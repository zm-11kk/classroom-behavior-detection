from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(r"F:\yolo_new_project\yolov8s.pt")
    results = model.train(
        data=r"F:\dataset5\datasets\datasets1\data.yaml",
        epochs=15,
        imgsz=640,
        batch=4,
        workers=2,
        amp=False,
        device=0,
        close_mosaic=3,
        lr0=0.001,
        lrf=0.01,
        patience=25,
        save=True,
        name="train_dataset5",
        exist_ok=False,
    )
#        Class            Images    Instances   Box(P        R      mAP50
#        all               1603      27072      0.794     0.749      0.789
#        Using_phone       346       1776      0.781      0.601      0.655
#        bend              286       1566      0.794      0.841      0.879
#        book              353       2388       0.78      0.784      0.816
#        bow_head          375       5151      0.867      0.933      0.931
#        hand-raising      893       2852      0.769      0.759      0.819
#        phone             236        658      0.813      0.436      0.496
#        raise_head        263       1560       0.86      0.899      0.928
#        reading           768       3454      0.665        0.7      0.734
#        sleep             105        212      0.966      0.858      0.898
#        turn_head         181        476      0.698      0.519       0.63
#        upright           386       5934      0.898      0.959      0.964
#        writing           286       1045      0.642      0.702      0.722
#Results saved to F:\yolo_new_project\runs\detect\train_dataset5
