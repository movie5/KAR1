Object_Labels = (
  'A',
  'B',
  'C',
  'D',
  'E',
)

"""
예시 label.txt

5 168 616 129 273
0 492 497 42 138
0 507 479 60 152
0 0 673 60 151
"""



class OilTankDetection(data.Dataset):

    def __init__(self,is_train=True):

        self.is_train = is_train
        self.opt = opt
        self.ids = []
        root = 'dataset'
        # train에 해당되는 이미지와 라벨
        if self.is_train:
            img_file = os.path.join(root,'oiltank_dataset/train_images/*')
            ano_file = os.path.join(root,'oiltank_dataset/train_labels/*')
            file_list = glob.glob(img_file)
            file_list_img = [file for file in file_list if file.endswith(".png")]

            for i in file_list_img:
                file_name = os.path.split(i)[1].split('.')[0]
                img = f"{root}/train_img/{file_name}.png"
                ano = f"{root}/train_label/{file_name}.json"
                
                if os.path.isfile(ano) and os.path.isfile(img):
                    self.ids.append((img, ano))
        else:
            img_file = os.path.join(root,'valid_images/*')
            file_list = glob.glob(img_file)
            file_list_img = [file for file in file_list if file.endswith(".png")]
            for i in file_list_img:
                self.ids.append((i))
        
    def __getitem__(self, index):
        if self.is_train:
            img_path, ano_path = self.ids[index]

            #cv2로 읽어야하나? 확인해보기
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            boxes, labels = self.get_annotations(ano_path)

            if self.is_train:
                image, boxes, labels = preproc_for_train(image, boxes, labels, opt.min_size, opt.mean)
                image = torch.from_numpy(image)

            target = np.concatenate([boxes, labels.reshape(-1,1)], axis=1)

            return image, target
        else: #오류 처리
            img_path = self.ids[index]
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            return image

"""
예시 label.txt

5 168 616 129 273
0 492 497 42 138
0 507 479 60 152
0 0 673 60 151
"""

    # json으로 읽은거를 저렇게 바꿔야함 그리고 우리는 8개 좌표임.
    def get_annotations(self, path):
        
        f = open(path, 'r')
        det = f.readlines()    #파일 리스트 불러오기
        boxes = []
        for d in det:
            obj = d.split(' ')      
            label = int(obj[0])  # 나중에 앞부분만 추출하는 코드 추가
          #아래 코드는 수정해야함)
            box = [float(obj[1]),float(obj[2]),float(obj[1])+float(obj[3]),float(obj[2])+float(obj[4])]
            boxes.append(box)
        return np.array(boxes), np.array(labels)
            

        

    def __len__(self):
        return len(self.ids)
        