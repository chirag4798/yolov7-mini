import cv2, os
import warnings
from tqdm import tqdm
from .yolov7 import YoloV7
warnings.simplefilter("ignore")


def plot_label(img, model, class_index, label_index, line_thickness=3):
    """
    Plot Text box on the original image.
    """
    x_offset, y_offset = 15, 45 * (label_index + 1)
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = model.colors[class_index]
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(f"{model.names[class_index]} Detected!", 0, fontScale=tl / 3, thickness=tf)[0]
    cv2.rectangle(img, (x_offset - 5, y_offset - 5), (x_offset + t_size[0] + 5, y_offset + t_size[1] + 5), color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, f"{model.names[class_index]} Detected!", (x_offset, y_offset + t_size[1]), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def process_image(image_path, output_dir, model):
    """
    Process one image_path and write visualization to output dir file.
    """
    print("\n===================================================== 💥 YOLOV7 💥 =====================================================\n")
    output = os.path.join(output_dir, os.path.basename(image_path).lower().replace(".jpg", "_visualized.jpg"))
    print(f"Input:\t'{image_path}'\nOutput:\t'{output}'")
    if os.path.isfile(output):
        print("Already processed! Skipping...")
        return

    in_frame         = cv2.imread(image_path)
    out_frame, boxes = model(in_frame)
    classes_deteced  = set([box["classes"] for box in boxes])
    for label_idx, cls in enumerate(classes_deteced):
        plot_label(out_frame, model, cls, label_idx)
    cv2.imwrite(output, out_frame)


def process_video(video_path, output_dir, model):
    """
    Process one video_path and write visualization to output mp4 file.
    """
    print("\n===================================================== 💥 YOLOV7 💥 =====================================================\n")
    output = os.path.join(output_dir, os.path.basename(video_path).replace(".mp4", "_visualized.mp4"))
    print(f"Input:\t'{video_path}'\nOutput:\t'{output}'")
    if os.path.isfile(output):
        print("Already processed! Skipping...")
        return

    # Stream settings
    cap     = cv2.VideoCapture(video_path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer  = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for _ in tqdm(range(nframes)):
        ret, in_frame = cap.read()
        if not ret:
            continue
        out_frame, boxes = model(in_frame)
        classes_deteced  = set([box["classes"] for box in boxes])
        for label_idx, cls in enumerate(classes_deteced):
            plot_label(out_frame, model, cls, label_idx)
        writer.write(out_frame)

    cap.release()
    writer.release()