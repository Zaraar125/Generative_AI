import os
import pandas as pd


def create_frames(train_dir,val_dir,test_dir):
    image_paths=os.listdir(train_dir+'\\photos')[:1500]
    complete_paths_1=[os.path.join(train_dir+'\\photos\\'+i) for i in image_paths]
    image_paths=os.listdir(train_dir+'\\sketches')[:1500]
    complete_paths_2=[os.path.join(train_dir+'\\sketches\\'+i) for i in image_paths]
    a=pd.DataFrame([complete_paths_1,complete_paths_2]).T
    a.rename(columns={0:'Faces',1:'Sketches'},inplace=True)

    image_paths=os.listdir(val_dir+'\\photos')
    complete_paths_1=[os.path.join(val_dir+'\\photos\\'+i) for i in image_paths]
    image_paths=os.listdir(val_dir+'\\sketches')
    complete_paths_2=[os.path.join(val_dir+'\\sketches\\'+i) for i in image_paths]
    b=pd.DataFrame([complete_paths_1,complete_paths_2]).T
    b.rename(columns={0:'Faces',1:'Sketches'},inplace=True)

    image_paths=os.listdir(test_dir+'\\photos')
    complete_paths_1=[os.path.join(test_dir+'\\photos\\'+i) for i in image_paths]
    image_paths=os.listdir(test_dir+'\\sketches')
    complete_paths_2=[os.path.join(test_dir+'\\sketches\\'+i) for i in image_paths]
    c=pd.DataFrame([complete_paths_1,complete_paths_2]).T
    c.rename(columns={0:'Faces',1:'Sketches'},inplace=True)
    return a,b,c
