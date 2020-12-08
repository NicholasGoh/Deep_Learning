import os, scipy, cv2, tqdm

def convert(path):
    os.makedirs(path.replace('annotations', 'yolo'), exist_ok=True)
    files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('.mat')]

    for i, file in tqdm.tqdm(enumerate(files)):
        data = scipy.io.loadmat(file, squeeze_me=True, simplify_cells=True)
        boxes = data['boxes']

        imgPath = file.replace('annotations', 'images').replace('.mat', '.jpg')
        yoloPath = imgPath.replace('images', 'yolo')
        txtPath = yoloPath.replace('.jpg', '.txt')
        os.system(f'touch {txtPath}')
        os.system(f'cp {imgPath} {yoloPath}')
        
        img = cv2.imread(imgPath)
        height, width, _ = img.shape
        for box in boxes:
            try:
                keys = list('abcd')
                values = [box[x].astype(int) for x in keys]
                allX = [x[1] for x in values]; allY = [x[0] for x in values]
            except Exception as e:
                continue

            topleft = (min(allX), max(allY))
            bottomright = (max(allX), min(allY))
            x = (bottomright[0] + topleft[0]) / 2 / width
            y = (topleft[1] + bottomright[1]) / 2 / height
            w = (bottomright[0] - topleft[0]) / width
            h = (topleft[1] - bottomright[1]) / height
            
            os.system(f'echo \'0 {x} {y} {w} {h}\' >> {txtPath}')
