from sklearn.cluster import KMeans
import cv2,os
import numpy as np
import pandas as pd
from PIL import Image
import random


def get_img(filepath):
    return cv2.imread(filepath)

def preprocess_img(img,shape=None):
    if shape:
        img = cv2.resize(img,shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape((-1,1))
    img = np.float32(img)
    img = np.array(img)

    return img

def process_img2(img,width,height,debug=0):
    img_by_rows = [img[i:i+width] for i in range(0,len(img),width)]
    
    if debug==1:
        print(len(img_by_rows)-height+1,len(img_by_rows),"height",height,"width",width)
        for i in range(0,len(img_by_rows)-height+1):
            print("into p2",i)
    img_by_section = [img_by_rows[i:i+height] for i in range(0,len(img_by_rows)-height+1)]
    img_tmp2 = []
    for l in img_by_section:
        l = np.array(l)
        img_tmp2.append(l.flatten())

    img_df = pd.DataFrame(img_tmp2, dtype=object)
    return img_df

def k_means(img_df,K_clusters):
    k_m = KMeans(n_clusters=K_clusters) 

    k_m.fit(img_df) 

    cluster_assignments = k_m.predict(img_df)

    return (cluster_assignments,k_m)

def get_legends(legend_folder):
    legends_fpath = []
    legends_name = []
    for (path,dirs,files) in os.walk(legend_folder):
        for f in files:
            if "legend" in f:
                legends_fpath.append(path+f)
                legends_name.append(f)


    INF = 1000000000
    legends = []
    legend_shape = [INF,INF]

    for fpath in legends_fpath:
        legends.append(cv2.imread(fpath))
        shape = legends[-1].shape[:2]
        if shape[0] < legend_shape[0]:
            legend_shape[0] = shape[0]
        if shape[1] < legend_shape[1]:
            legend_shape[1] = shape[1]

    # resize legends to have the same shape
    for i in range(len(legends)):
        legends[i] = cv2.resize(legends[i], tuple(legend_shape))
        # legends[i] = cv2.cvtColor(legends[i], cv2.COLOR_BGR2GRAY)
        legends[i] = preprocess_img(legends[i])
        # print("Legend shape",len(legends[0]))
    
    return (legend_shape,legends,legends_name)

#PLOT
def plot_res_cluster(k_clusters,clusters):
    colors = []
    for i in range(k_clusters):
        colors.append(random.randint(0,255))

    img_save = []
    width_img_save = 61
    for i in range(len(clusters)):
        img_save.append([])
        for j in range(width_img_save):
            img_save[i].append(colors[clusters[i]])

    array = np.array(img_save, dtype=np.uint8)

    new_image = Image.fromarray(array)
    new_image.save('new.png')


def cluster_log(PATH_LOG, PATH_LEGEND_FOLDER):
    #PREPROCESS
    log_img = get_img(PATH_LOG)
    
    (legend_shape,legends,legends_name) = get_legends(PATH_LEGEND_FOLDER)
    
    k_clusters = len(legends)+2
    
    log_pre = preprocess_img(log_img,(legend_shape[1],log_img.shape[:2][0]))
    
    log_df = process_img2(log_pre, legend_shape[1], legend_shape[0])
    
    legends_lst = []
    for i,legend in enumerate(legends):
        legends_lst.append((process_img2(legend,legend_shape[1], legend_shape[0], debug=0),legends_name[i]))
    
    #COMPUTE
    #Train on legends, predict on log BEST RESULTS
    
    frame = pd.concat([leg_df for (leg_df,name) in legends_lst])
    
    for (leg_df,name) in legends_lst:
        print(leg_df.shape)
    
    print(frame.shape)
    print(log_df.shape)
    
    #k_clusters = len(legends_lst) -> because n_sample should be >= n_clusters
    (clusters,k_m) = k_means(frame,len(legends_lst))
    
    cluster_and_legend = [x for x in zip(clusters,[name[7:-4] for _,name in legends_lst if len(name) > 11])]
    
    print(cluster_and_legend)
    
    clusters = k_m.predict(log_df)
    
    plot_res_cluster(len(legends_lst),clusters)
    
    cl = list(clusters)
    
    dict_data = {"total":0}
    for cluster_id,legend in cluster_and_legend:
        dict_data[legend] = cl.count(cluster_id)
        dict_data['total'] += cl.count(cluster_id)
    

    for key in dict_data:
        if key != 'total':
            print(round(100*dict_data[key]/dict_data['total'],2),key)
    return dict_data
