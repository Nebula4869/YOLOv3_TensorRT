from post_processor import predict_transform, write_results
from onnx_to_tensorrt import get_engine
import numpy as np
import common
import torch
import time
import cv2


def main(input_size):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    class_names = [line.rstrip('\n') for line in open('./coco.names')]
    num_classes = len(class_names)

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = './models/yolov3-%d.onnx' % input_size
    engine_file_path = "./models/yolov3-%d.trt" % input_size

    # Output shapes expected by the post-processor
    output_shapes = [(1, 255, input_size // 32, input_size // 32),
                     (1, 255, input_size // 16, input_size // 16),
                     (1, 255, input_size // 8, input_size // 8)]

    # Anchors expected by the post-processor
    yolo_anchors = [[(116, 90), (156, 198), (373, 326)],
                    [(30, 61), (62, 45), (59, 119)],
                    [(10, 13), (16, 30), (33, 23)]]

    torch.cuda.FloatTensor()
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = (np.expand_dims(np.transpose(cv2.resize(frame, (input_size, input_size)), [2, 0, 1]), axis=0) / 255.).astype(np.float32)
            inference_start = time.time()
            inputs[0].host = np.array(image, dtype=np.float32, order='C')
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            inference_time = time.time() - inference_start
            yolo_start = time.time()
            # Do yolo_layer with pytorch
            write = False
            for output, shape, anchors in zip(trt_outputs, output_shapes, yolo_anchors):
                output = output.reshape(shape)
                trt_output = torch.from_numpy(output).cuda()
                trt_output = trt_output.data
                trt_output = predict_transform(trt_output, input_size, anchors, num_classes, True)
                if type(trt_output) == int:
                    continue
                if not write:
                    detections = trt_output
                    write = True
                else:
                    detections = torch.cat((detections, trt_output), 1)

            dets = write_results(detections, 0.5, num_classes, nms=True, nms_conf=0.45)
            if type(dets) == int:
                dets = []
            else:
                dets = dets.cpu().numpy()
            yolo_time = time.time() - yolo_start

            for i in range(len(dets)):
                left = max(0, int(dets[i][1] * 1920 / input_size))
                top = max(0, int(dets[i][2] * 1080 / input_size))
                right = min(frame.shape[1], int(dets[i][3] * 1920 / input_size))
                bottom = min(frame.shape[0], int(dets[i][4] * 1080 / input_size))
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0))
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0))
                cv2.putText(frame, '%s: %.4f' % (class_names[int(dets[i][7])], dets[i][5]), (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

            cv2.putText(frame, 'Inference Time: %.2fms' % (inference_time * 1000), (50, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
            cv2.putText(frame, 'YOLO Time: %.2fms' % (yolo_time * 1000), (50, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
            cv2.imshow('', frame)
            if cv2.waitKey(1) == 27:
                break


if __name__ == '__main__':
    main(input_size=608)
