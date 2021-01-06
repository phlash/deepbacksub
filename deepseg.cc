/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
	http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// tested against tensorflow lite v2.1.0 (static library)

#include <unistd.h>
#include <signal.h>
#include <execinfo.h>
#include <cstdio>

#include <opencv2/opencv.hpp>

#include "loopback.h"
#include "capture.h"
#include "inference.h"
#include "dlibhog.h"


#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
	fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }

// crash dumper
void trap(int sig) {
#define MAXOOPS 20
	void *oops[MAXOOPS];
	fprintf(stderr, "SIGNAL:%d\n", sig);
	int n=backtrace(oops, MAXOOPS);
	backtrace_symbols_fd(oops, n, 2);
	exit(1);
}

// deeplabv3 classes
std::vector<std::string> labels = { "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv" };

typedef struct {
	capinfo_t *pcap;
	capinfo_t *pbkg;
	cv::Mat bg;
	cv::Mat mask;
	int lbfd;
	int outw, outh;
	int debug;
	bool done;
	pthread_mutex_t lock;
} frame_ctx_t;

// Process an incoming raw video frame
bool process_frame(cv::Mat *cap, void *ctx) {
	frame_ctx_t *pfr = (frame_ctx_t *)ctx;
	// grab next available background frame (if video)
	if (pfr->pbkg!=NULL) {
		capture_frame(pfr->pbkg, pfr->bg);
		// resize to output if required
		if (pfr->bg.cols != pfr->outw || pfr->bg.rows != pfr->outh)
			cv::resize(pfr->bg,pfr->bg,cv::Size(pfr->outw,pfr->outh));
	}
	// otherwise assume pfr->bg is a suitable static image..

	// resize capture frame if required
	if (cap->cols != pfr->outw || cap->rows != pfr->outh)
		cv::resize(*cap,*cap,cv::Size(pfr->outw,pfr->outh));

	// alpha blend cap and background images using mask, adapted from:
	// https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/
	cv::Mat out = cv::Mat::zeros(cap->size(), cap->type());
	uint8_t *optr = (uint8_t*)out.data;
	pthread_mutex_lock(&pfr->lock);     // (lock to protect access to mask.data)
	uint8_t *rptr = (uint8_t*)cap->data;
	uint8_t *bptr = (uint8_t*)pfr->bg.data;
	float   *aptr = (float*)pfr->mask.data;
	int npix = cap->rows * cap->cols;
	for (int pix=0; pix<npix; ++pix) {
		// blending weights
		float rw=*aptr, bw=1.0-rw;
		// blend each channel byte
		*optr = (uint8_t)( (float)(*rptr)*rw + (float)(*bptr)*bw ); ++rptr; ++bptr; ++optr;
		*optr = (uint8_t)( (float)(*rptr)*rw + (float)(*bptr)*bw ); ++rptr; ++bptr; ++optr;
		*optr = (uint8_t)( (float)(*rptr)*rw + (float)(*bptr)*bw ); ++rptr; ++bptr; ++optr;
		++aptr;
	}
	pthread_mutex_unlock(&pfr->lock);

	// write frame to v4l2loopback
	cv::Mat yuv;
	cv::cvtColor(out,yuv,CV_BGR2YUV_I420);
	int framesize = yuv.step[0]*yuv.rows;
	while (framesize > 0) {
		int ret = write(pfr->lbfd,yuv.data,framesize);
		if (ret <= 0)
			return false;
		framesize -= ret;
	}

	char ti[64];
	if (pfr->debug > 2) {
		sprintf(ti, "cap: %dx%d/%d", cap->cols, cap->rows, cap->type());
		cv::imshow(ti,*cap);
		sprintf(ti, "bg: %dx%d/%d", pfr->bg.cols, pfr->bg.rows, pfr->bg.type());
		cv::imshow(ti,pfr->bg);
		sprintf(ti, "mask: %dx%d/%d", pfr->mask.cols, pfr->mask.rows, pfr->mask.type());
		cv::imshow(ti,pfr->mask);
	}
	if (pfr->debug > 1) {
		sprintf(ti, "out: %dx%d/%d", out.cols, out.rows, out.type());
		cv::imshow(ti,out);
		if (cv::waitKey(1) == 'q') pfr->done = true;
	}
	return true;
}

int main(int argc, char* argv[]) {

	printf("deepseg v0.2.1\n");
	printf("(c) 2021 by floe@butterbrot.org - https://github.com/floe/deepseg\n");
	printf("(c) 2021 by phil.github@ashbysoft.com - https://github.com/phlash/deepseg\n");

	signal(SIGSEGV, trap);
	signal(SIGABRT, trap);
	int debug  = 0;
	int threads= 2;
	int width  = 640;
	int height = 480;
	const char *back = "images/background.png";
	const char *vcam = "/dev/video0";
	const char *ccam = "/dev/video1";

	bool usehog = false;
	const char* modelname = "models/segm_full_v679.tflite";

	for (int arg=1; arg<argc; arg++) {
		if (strncmp(argv[arg], "-?", 2)==0) {
			fprintf(stderr, "usage: deepseg [-?] [-d] [-c <capture:%s>] [-v <vcam:%s>] [-w <width:%d>] [-h <height:%d>] "
							"[-t <threads:%d>] [-b <%s>] [-m <%s>] [-g (use hoG):%s]\n",
							ccam,vcam,width,height,threads,back,modelname,usehog?"true":"false");
			exit(0);
		} else if (strncmp(argv[arg], "-d", 2)==0) {
			++debug;
		} else if (strncmp(argv[arg], "-g", 2)==0) {
			usehog = true;
		} else if (strncmp(argv[arg], "-v", 2)==0) {
			vcam = argv[++arg];
		} else if (strncmp(argv[arg], "-c", 2)==0) {
			ccam = argv[++arg];
		} else if (strncmp(argv[arg], "-b", 2)==0) {
			back = argv[++arg];
		} else if (strncmp(argv[arg], "-m", 2)==0) {
			modelname = argv[++arg];
		} else if (strncmp(argv[arg], "-w", 2)==0) {
			sscanf(argv[++arg], "%d", &width);
		} else if (strncmp(argv[arg], "-h", 2)==0) {
			sscanf(argv[++arg], "%d", &height);
		} else if (strncmp(argv[arg], "-t", 2)==0) {
			sscanf(argv[++arg], "%d", &threads);
		}
	}
	printf("debug:  %d\n", debug);
	printf("ccam:   %s\n", ccam);
	printf("vcam:   %s\n", vcam);
	printf("width:  %d\n", width);
	printf("height: %d\n", height);
	printf("back:   %s\n", back);
	printf("threads:%d\n", threads);
	printf("model:  %s\n", modelname);
	printf("usehog: %d\n", usehog);

	// context data shared with callback
	frame_ctx_t fctx;
	fctx.lock = PTHREAD_MUTEX_INITIALIZER;
	fctx.done = false;
	fctx.debug = debug;
	fctx.outw = width;
	fctx.outh = height;
	// open loopback virtual camera stream, always with YUV420p output
	fctx.lbfd = loopback_init(vcam,width,height,debug);
	// open capture device stream, pass in/out expected/actual size
	int capw = width, caph = height, rate;
	fctx.pcap = capture_init(ccam, &capw, &caph, &rate, debug);
	TFLITE_MINIMAL_CHECK(fctx.pcap!=NULL);
	printf("stream info: %dx%d @ %dfps\n", capw, caph, rate);

	// check background file extension (yeah, I know) to spot videos..
	fctx.pbkg = NULL;
	int bkgw = width, bkgh = height;
	char *dot = rindex((char*)back, '.');
	if (dot!=NULL &&
		(strcasecmp(dot, ".png")==0 ||
		 strcasecmp(dot, ".jpg")==0 ||
		 strcasecmp(dot, ".jpeg")==0)) {
		// read background into raw BGR24 format, resize to output
		fctx.bg = cv::imread(back);
		cv::resize(fctx.bg,fctx.bg,cv::Size(width,height));
	} else {
		// assume video background..start capture
		fctx.pbkg = capture_init(back, &bkgw, &bkgh, &rate, debug);
		TFLITE_MINIMAL_CHECK(fctx.pbkg!=NULL);
	}

	// Are we flowing or hogging?
	hoginfo_t *phg = NULL;
	tfinfo_t *ptf = NULL;
	cv::Mat input;
	cv::Mat output;
	float ratio = 1.0f;
	if (usehog) {
		// Load HOG
		phg = hog_init(debug);
	} else {
		// Load TF model
		ptf = tf_init(modelname, threads, debug);

		// wrap input and output tensor with cv::Mat
		tfbuffer_t *tbuf = tf_get_buffer(ptf, TFINFO_BUF_IN);
		input = cv::Mat(tbuf->h, tbuf->w, CV_32FC(tbuf->c), tbuf->data);
		delete tbuf;
		tbuf = tf_get_buffer(ptf, TFINFO_BUF_OUT);
		output = cv::Mat(tbuf->h, tbuf->w, CV_32FC(tbuf->c), tbuf->data);
		delete tbuf;
		ratio = (float)input.rows/(float) input.cols;
		printf("model rows=%d, cols=%d ratio=%f\n", input.rows, input.cols, ratio);
	}

	// initialize mask and square ROI in center
	cv::Rect roidim = cv::Rect((width-height/ratio)/2,0,height/ratio,height);
	cv::Mat mask = cv::Mat::zeros(height,width,CV_32FC1);
	cv::Mat mroi = mask(roidim);
	printf("roidim(x,y,w,h)=(%d,%d,%d,%d)\n",roidim.x,roidim.y,roidim.width,roidim.height);
	mask.copyTo(fctx.mask);

	// erosion/dilation elements
	cv::Mat element3 = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size(3,3) );
	cv::Mat element7 = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size(7,7) );
	cv::Mat element11 = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size(11,11) );

	// label number of "person" for DeepLab v3+ model
	const int cnum = labels.size();
	const int pers = std::find(labels.begin(),labels.end(),"person") - labels.begin();

	// attach input frame callback
	capture_setcb(fctx.pcap, process_frame, &fctx);

	// stats
	int64 es = cv::getTickCount();
	int64 e1 = es;
	int64 fr = 0;
	int64 lcap = 0;
	while (!fctx.done) {

		// wait for next capture frame (we might be quicker than input rate now!)
		while (lcap==capture_count(fctx.pcap)) {
			struct timespec ts = { 0, 1000000 }; // 1ms
			clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, NULL);
		}
		lcap = capture_count(fctx.pcap);

		// grab last captured frame
		cv::Mat cap;
		capture_frame(fctx.pcap, cap);

		// HOG or TF sir?
		if (usehog) {
			// Resize to output if required
			if (cap.cols != fctx.outw || cap.rows != fctx.outh)
				cv::resize(cap,cap,cv::Size(fctx.outw,fctx.outh));

			// Run HOG to rough mask
			TFLITE_MINIMAL_CHECK(hog_faces(phg, cap, output));

			// smooth mask..
			if (!output.empty() && getenv("DEEPSEG_NOBLUR")==NULL)
				cv::blur(output,mask,cv::Size(7,7));
		} else {
			// map ROI
			cv::Mat roi = cap(roidim);
			// convert BGR to RGB, resize ROI to input size
			cv::Mat in_u8_rgb, in_resized;
			cv::cvtColor(roi,in_u8_rgb,CV_BGR2RGB);
			// TODO: can convert directly to float?
			cv::resize(in_u8_rgb,in_resized,cv::Size(input.cols,input.rows));
			if (debug > 2) cv::imshow("input",in_resized);

			// convert to float and normalize values to [-1;1]
			in_resized.convertTo(input,CV_32FC3,1.0/128.0,-1.0);

			// Run inference
			TFLITE_MINIMAL_CHECK(tf_infer(ptf));

			// create Mat for small mask
			cv::Mat ofinal(output.rows,output.cols,CV_32FC1);
			float* tmp = (float*)output.data;
			float* out = (float*)ofinal.data;

			// find class with maximum probability
			if (strstr(modelname, "deeplab")) {
				for (unsigned int n = 0; n < output.total(); n++) {
					float maxval = -10000; int maxpos = 0;
					for (int i = 0; i < cnum; i++) {
						if (tmp[n*cnum+i] > maxval) {
							maxval = tmp[n*cnum+i];
							maxpos = i;
						}
					}
					// set mask to 1.0 where class == person
					out[n] = (maxpos==pers ? 1.0 : 0);
				}
   			} else if (strstr(modelname,"body-pix")) {
				for (unsigned int n = 0; n < output.total(); n++) {
					if (tmp[n] < 0.65) out[n] = 0; else out[n] = 1.0;
				}
			} else if (strstr(modelname,"segm_")) {
				// Google Meet segmentation network
					/* 256 x 144 x 2 tensor for the full model or 160 x 96 x 2
					 * tensor for the light model with masks for background
					 * (channel 0) and person (channel 1) where values are in
					 * range [MIN_FLOAT, MAX_FLOAT] and user has to apply
					 * softmax across both channels to yield foreground
					 * probability in [0.0, 1.0]. */
				for (unsigned int n = 0; n < output.total(); n++) {
					float exp0 = expf(tmp[2*n  ]);
					float exp1 = expf(tmp[2*n+1]);
					float p0 = exp0 / (exp0+exp1);
					float p1 = exp1 / (exp0+exp1);
					if (p0 < p1) out[n] = 1.0; else out[n] = 0;
				}
			}
			if (debug > 2) cv::imshow("ofinal",ofinal);

			// denoise, close & open with small then large elements, adapted from:
			// https://stackoverflow.com/questions/42065405/remove-noise-from-threshold-image-opencv-python
			if (getenv("DEEPSEG_NODENOISE")==NULL) {
				cv::morphologyEx(ofinal,ofinal,CV_MOP_CLOSE,element3);
				cv::morphologyEx(ofinal,ofinal,CV_MOP_OPEN,element3);
				cv::morphologyEx(ofinal,ofinal,CV_MOP_CLOSE,element7);
				cv::morphologyEx(ofinal,ofinal,CV_MOP_OPEN,element7);
				cv::dilate(ofinal,ofinal,element7);
			}
			// smooth mask edges
			if (getenv("DEEPSEG_NOBLUR")==NULL)
				cv::blur(ofinal,ofinal,cv::Size(7,7));
			// scale up into full-sized mask
			cv::resize(ofinal,mroi,cv::Size(mroi.cols,mroi.rows));

		}
		// update mask for render thread (under lock)
		pthread_mutex_lock(&fctx.lock);
		mask.copyTo(fctx.mask);
		pthread_mutex_unlock(&fctx.lock);
		++fr;

		if (!debug) { printf("."); fflush(stdout); continue; }

		int64 e2 = cv::getTickCount();
		float el = (e2-e1)/cv::getTickFrequency();
		float t = (e2-es)/cv::getTickFrequency();
		e1 = e2;
		int64 rcnt = capture_count(fctx.pcap);
		int64 bcnt = fctx.pbkg!=NULL ? capture_count(fctx.pbkg) : 0;
		printf("\relapsed:%0.3f gr=%ld gps:%3.1f br=%ld fr=%ld fps:%3.1f   ",
			el, rcnt, rcnt/t, bcnt, fr, fr/t);
		fflush(stdout);
	}
	capture_stop(fctx.pcap);
	if (fctx.pbkg!=NULL)
		capture_stop(fctx.pbkg);

	return 0;
}

