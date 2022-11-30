import numpy as np
import json
import os
import numpy as np
# 폴더 경로 입력하면 이 폴더에 있는 json file 목록들을 list에 저장하고 이 list return 하는 function
def get_json_list(path_dir):
    file_list=os.listdir(path_dir) # dir 경로(path_dir) 입력하면 이 디렉토리에 있는 파일들 저장 
    list1=[file for file in file_list if file.endswith('.json')] # 디렉토리에 있는 파일들(file_list) 중에서 json 파일만 저장
    return list1

# polygon 형식의 데이터에서 정보 얻어오는 function
def get_annotation_pol(list1, dict,id):
    value=np.array(list(dict.values())) # x, y좌표(value)들 가져오기
    px=np.zeros(int((value.shape[0])/2)) # x 좌표만 모아둔거
    py=np.zeros(int((value.shape[0])/2)) # y 좌표만 모아둔거
    for k in range(0,value.shape[0]):
        if k%2==0:
            px[k//2]=value[k]
        else:
            py[k//2]=value[k]
        
    obj = {
            "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
            "bbox_mode": 0,
            "segmentation": [[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]],
            "category_id": id,
            }
    list1.append(obj)
    
    
    
# box 형식의 데이터에서 정보 얻어오는 function
def get_annotation_box(list1, loc_dict,id):
    xmin=loc_dict.get("x")
    ymin=loc_dict.get("y") # x, y 좌표 가져오기
    width=loc_dict.get("width")
    height=loc_dict.get("height")
    xmax=xmin+width
    ymax=ymin+height

    obj = {
            "bbox": [xmin, ymin, xmax, ymax],
            "bbox_mode": 0,
           "segmentation":[[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]],
            "category_id": id,
           
            }

    list1.append(obj)
    
# 얻어야하는 카테고리별(경기장, 3점라인, 페인트존, 집단행동참여자, 골대)별 id 부여
# 각 카테고리마다 get_annotation_box(pol)에서 정보 추출
def get_info(dataset_dicts, imgs_anns,image_id,img_dir):
    
    env=imgs_anns["labelinginfo_scene representation"]
    filename = os.path.join("/local_datasets/detectron2/basketball/annotations/images", imgs_anns["metaData"]["농구메타데이터"]+".jpg")
    if not os.path.isfile(filename):
        print(filename)
        print("Not img file")
        exit(0)
    record = {} # 하나의 json 파일에 들어있는 정보들 저장하기 위한 dictionary
    objs=[]
    record["file_name"]=filename
    record["image_id"]=image_id
    record["height"] = imgs_anns["imageinfo"]["height"]
    record["width"] = imgs_anns["imageinfo"]["width"]

    for h in env:
        dict=env[h]
        if(h=="환경"): # 환경인 경우
            for i in dict: # 경기장인 경우
                if(i=="경기장"):
                    for a in range(0,len(dict[i]["location"])):
                        get_annotation_pol(objs, dict[i]["location"][a],0)
                    record["annotations"]=objs           
                else: # 경기라인인 경우
                    for j in dict[i]: # 3점라인, 페인트존
                        for a in range(0,len(dict[i][j]["location"])):
                            if(j=="3점라인"): # 3점라인
                                get_annotation_pol(objs, dict[i][j]["location"][a],1)
                            else: #페인트존
                                get_annotation_pol(objs, dict[i][j]["location"][a],2)
                    record["annotations"]=objs
        elif(h=="집단행동참여자"):
            get_annotation_box(objs, dict[0]["location"],3)
        elif(h=="골대"): # 골대
            for i in range(0,len(dict["location"])): # 골대 2개인거 이용
                get_annotation_box(objs, dict["location"][i],4)
        else: # 경기도구
            get_annotation_box(objs, dict["location"],5)
    dataset_dicts.append(record)
# 앞의 함수들 통해 json별로 정보 추출하고 이를 dataset_dicts2라는 list에 저장
def get_sports_dicts(dir):
    json_list=get_json_list(dir) # 디렉토리에 있는 json 파일 불러오기
    dataset_dicts2=[] # json 별 정보 저장한 dictionary 보관하기 위한 list
    for idx, i in enumerate(json_list):
        with open(os.path.join(dir,i),'r',encoding='UTF8') as f:
            imgs_anns2=json.load(f)
        get_info(dataset_dicts2,imgs_anns2,idx,dir) # json별 정보 얻기
    
    return dataset_dicts2