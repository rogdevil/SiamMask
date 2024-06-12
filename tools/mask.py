import glob
from tools.test import *
import os
import time
from types import SimpleNamespace
import shutil


def loadSiam(resume, base_path, config, cpu):
    print(config)
    args = SimpleNamespace(
        **{"resume": resume, "config": config, "base_path": base_path, "cpu": cpu}
    )  # replace with your actual config file
    cfg = load_config(
        args=args,
    )  # assuming you have a function that loads the config
    from experiments.siammask_sharp.custom import Custom

    siammask = Custom(anchors=cfg["anchors"])
    if args.resume:
        assert isfile(args.resume), "Please download {} first.".format(args.resume)
        siammask = load_pretrain(siammask, args.resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    siammask.eval().to(device)
    return siammask, cfg, device


def maskUsingSiam(video):
    time_start = time.time()
    cap = cv2.VideoCapture(video)
    projectDir = os.path.dirname(__file__).split("/tools")[0]

    siammask, cfg, device = loadSiam(
        os.path.join(projectDir, "experiments/siammask_sharp", "SiamMask_DAVIS.pth"),
        "data/tennis",
        os.path.join(projectDir, "experiments/siammask_sharp", "config_davis.json"),
        None,
    )
    try:
        shutil.rmtree("frames")
    except FileNotFoundError:
        pass

    os.makedirs("frames")
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    objectSelected = False
    # Start converting the video
    print("processing video", video_length)
    while cap.isOpened():
        ret, frame = cap.read()
        if count <= video_length - 1:
            count += 1
            cv2.imwrite(f"frames/{count}.jpg", frame)
        else:
            break
    print("processing finished")
    cap.release()

    frames_folder_path = "frames"
    image_files = [f for f in os.listdir(frames_folder_path) if f.endswith(".jpg")]
    targetFrame = 0
    for file in image_files:
        file_path = os.path.join(frames_folder_path, file)
        img = cv2.imread(file_path)
        if not objectSelected:
            cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
            try:
                init_rect = cv2.selectROI("SiamMask", img, False, False)
                x, y, w, h = init_rect
                if w > 1 and h > 1:
                    objectSelected = True
                    targetFrame = int(file_path.split("/")[-1].replace(".jpg", ""))
            except:
                print("no object selected")
                exit()
    toc = 0
    for file in image_files:
        tic = cv2.getTickCount()
        file_path = os.path.join(frames_folder_path, file)
        f = int(file_path.split("/")[-1].replace(".jpg", ""))
        im = cv2.imread(file_path)
        if f == targetFrame:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(
                im, target_pos, target_sz, siammask, cfg["hp"], device=device
            )  # init tracker
        elif f > targetFrame:  # tracking
            state = siamese_track(
                state, im, mask_enable=True, refine_enable=True, device=device
            )  # track
            location = state["ploygon"].flatten()
            mask = state["mask"] > state["p"].seg_thr

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            cv2.polylines(
                im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3
            )
            cv2.imshow("SiamMask", im)
            key = cv2.waitKey(1)
            if key > 0:
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print(
        "SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)".format(
            toc, fps
        )
    )


if __name__ == "__main__":
    # projectDir = os.path.dirname(__file__).split("/tools")[0]

    # siammask = loadSiam(
    #     os.path.join(projectDir, "experiments/siammask_sharp", "SiamMask_DAVIS.pth"),
    #     "data/tennis",
    #     os.path.join(projectDir, "experiments/siammask_sharp", "config_davis.json"),
    #     None,
    # )

    maskUsingSiam(
        "/Users/office/Downloads/videoplayback-small (online-video-cutter.com).mp4"
    )
