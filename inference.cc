// Wrapper for tensorflow inference processing
//
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>

#include "inference.h"
#include "transpose_conv_bias.h"

using namespace tflite;

#define ASSERT_OR_NULL(x) { if (!(x)) return NULL; }

struct _tfinfo_t {
	std::unique_ptr<tflite::FlatBufferModel> model;
	std::unique_ptr<Interpreter> interpreter;
	int debug;
};

tfinfo_t *tf_init(const char *modelname, int threads, int debug) {
	// Allocate info block
	tfinfo_t *ptf = new tfinfo_t;
	ptf->debug = debug;

	// Load model
	ptf->model = tflite::FlatBufferModel::BuildFromFile(modelname);
	ASSERT_OR_NULL(ptf->model != nullptr);

	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	// custom op for Google Meet network
	resolver.AddCustom("Convolution2DTransposeBias", mediapipe::tflite_operations::RegisterConvolution2DTransposeBias());
	InterpreterBuilder builder(*(ptf->model), resolver);
	builder(&ptf->interpreter);
	ASSERT_OR_NULL(ptf->interpreter != nullptr);

	// Allocate tensor buffers.
	ASSERT_OR_NULL(ptf->interpreter->AllocateTensors() == kTfLiteOk);

	// set interpreter params
	ptf->interpreter->SetNumThreads(threads);
	ptf->interpreter->SetAllowFp16PrecisionForFp32(true);

	return ptf;
}

tfbuffer_t *tf_get_buffer(tfinfo_t *ptf, int which) {
	int tnum = (0==which) ? ptf->interpreter->inputs()[0] : ptf->interpreter->outputs()[0];
	TfLiteType t_type = ptf->interpreter->tensor(tnum)->type;
	ASSERT_OR_NULL(t_type == kTfLiteFloat32);

	TfLiteIntArray* dims = ptf->interpreter->tensor(tnum)->dims;
	if (ptf->debug) for (int i = 0; i < dims->size; i++) printf("tensor #%d: %d\n",tnum,dims->data[i]);
	ASSERT_OR_NULL(dims->data[0] == 1);

	tfbuffer_t *pbuf = new tfbuffer_t;
	pbuf->h = dims->data[1];
	pbuf->w = dims->data[2];
	pbuf->c = dims->data[3];
	pbuf->data = ptf->interpreter->typed_tensor<float>(tnum);
	ASSERT_OR_NULL(pbuf->data != nullptr);
	return pbuf;
}

bool tf_infer(tfinfo_t *ptf) {
	return (ptf->interpreter->Invoke() == kTfLiteOk);
}

void tf_stop(tfinfo_t *ptf) {
}
