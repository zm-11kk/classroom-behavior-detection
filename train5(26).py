from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(r"F:\yolo_new_project\yolo26s.pt")
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
        name="train_dataset5_yolo26",
        exist_ok=False,
    )
#   Class         Images   Instances  Box(P      R          mAP50
#   all           1603      27072     0.797      0.751      0.794
#   sing_phone    346       1776      0.749      0.623      0.655
#   bend          286       1566      0.789      0.843      0.871
#   book          353       2388      0.811      0.748      0.816
#   bow_head      375       5151      0.869      0.936      0.929
#   hand-raising  893       2852      0.779      0.749      0.82
#   phone         236        658      0.834      0.442      0.525
#   raise_head    263       1560      0.868      0.903      0.929
#   reading       768       3454      0.682      0.702      0.745
#   sleep         105        212      0.929      0.866      0.899
#   turn_head     181        476      0.695      0.536      0.637
#   upright       386       5934      0.898      0.958      0.963
#   writing       286       1045      0.655      0.705      0.734
#Results saved to F:\yolo_new_project\runs\detect\train_dataset5_yolo26