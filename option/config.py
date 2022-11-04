import sys
import numpy as np

params_name=["start_x","start_y","end_x","end_y","middle_x","middle_y","thickness","RGB","alpha"]
epoch=2

class Options(object):
    make_list=["Foundation","Lipstick","Eye","Face"]
    name='test_ref'
    device='cuda'
    make={}
    make["Foundation"]={}
    make["Foundation"]["epoch"]=epoch
    make["Foundation"]["optimizer"]="Foundation"
    make["Foundation"]["Params"]={}
    make["Foundation"]["Params"]["RGB"]=[0.5,0.5,0.5]
    make["Foundation"]["Params"]["alpha"]=[0.9]


    make["Lipstick"]={}
    make["Lipstick"]["epoch"]=epoch
    make["Lipstick"]["optimizer"]="Lipstick"
    make["Lipstick"]["Params"]={}
    make["Lipstick"]["Params"]["RGB"]=[0.7,0.01,0.01]
    make["Lipstick"]["Params"]["alpha"]=[0.7]

    make["Eye"]={}
    make["Eye"]["epoch"]=epoch
    make["Eye"]["Params"]={}
    make["Eye"]["optimizer"]="Eye"

    RightEyeLineParams=[]
    RightEyeLineParams.append([[0.5],[0.33],[0.5],[0.67],[0.3],[0.5],[0.2],[0.02,0.02,0.1],[0.91]])
    RightEyeLineParams.append([[0.37],[0.58],[0.67],[0.8],[0.5],[0.67],[0.3],[0.02,0.02,0.1],[0.91]])
    RightEyeLineParam=np.array(RightEyeLineParams).T.tolist()
    RightEyeLineParam=dict(zip(params_name,RightEyeLineParam))
    RightEyeLineParam["useMirror"]=False
    RightEyeLineParam["useMulti"]=False
    RightEyeLineParam["renderMode"]='pen'
    RightEyeLineParam["right"]=True
    make["Eye"]["Params"]["RightEyeLine"]=RightEyeLineParam


    RightEyeParams=[]
    RightEyeParams.append([[0.5],[0.33],[0.5],[0.67],[0.3],[0.5],[0.75],[0.02,0.02,0.1],[0.91]])
    RightEyeParams.append([[0.5],[0.33],[0.5],[0.67],[0.3],[0.5],[0.2],[0.02,0.02,0.1],[0.91]])
    RightEyeParams.append([[0.5],[0.33],[0.5],[0.67],[0.3],[0.3],[0.25],[0.02,0.02,0.1],[0.91]])
    RightEyeParam=np.array(RightEyeParams).T.tolist()
    RightEyeParam=dict(zip(params_name,RightEyeParam))
    RightEyeParam["useMirror"]=False
    RightEyeParam["useMulti"]=False
    RightEyeParam["renderMode"]='powder'
    RightEyeParam["right"]=True
    make["Eye"]["Params"]["RightEye"]=RightEyeParam

    LeftEyeLineParams=[]
    LeftEyeLineParams.append([[0.5],[0.33],[0.5],[0.67],[0.3],[0.5],[0.2],[0.02,0.02,0.1],[0.91]])
    LeftEyeLineParams.append([[0.37],[0.58],[0.67],[0.8],[0.5],[0.67],[0.3],[0.02,0.02,0.1],[0.91]])
    LeftEyeLineParam=np.array(LeftEyeLineParams).T.tolist()
    LeftEyeLineParam=dict(zip(params_name,LeftEyeLineParam))
    LeftEyeLineParam["useMirror"]=False
    LeftEyeLineParam["useMulti"]=False
    LeftEyeLineParam["renderMode"]='pen'
    LeftEyeLineParam["right"]=False
    make["Eye"]["Params"]["LeftEyeLine"]=LeftEyeLineParam

    LeftEyeParams=[]
    LeftEyeParams.append([[0.5],[0.33],[0.5],[0.67],[0.3],[0.5],[0.75],[0.02,0.02,0.1],[0.91]])
    LeftEyeParams.append([[0.5],[0.33],[0.5],[0.67],[0.3],[0.5],[0.2],[0.02,0.02,0.1],[0.91]])
    LeftEyeParams.append([[0.5],[0.33],[0.5],[0.67],[0.3],[0.3],[0.25],[0.02,0.02,0.1],[0.91]])
    LeftEyeParam=np.array(LeftEyeParams).T.tolist()
    LeftEyeParam=dict(zip(params_name,LeftEyeParam))
    LeftEyeParam["useMirror"]=False
    LeftEyeParam["useMulti"]=False
    LeftEyeParam["renderMode"]='powder'
    LeftEyeParam["right"]=False
    make["Eye"]["Params"]["LeftEye"]=LeftEyeParam


    make["Face"]={}
    make["Face"]["epoch"]=epoch
    make["Face"]["Params"]={}
    make["Face"]["optimizer"]="Face"
    
    MirrorNoseParams=[]
    MirrorNoseParams.append([[0.25],[0.4],[0.45],[0.4],[0.35],[0.4],[0.2],[0.2,0.2,0.2],[0.91]])
    MirrorNoseParam=np.array(MirrorNoseParams).T.tolist()
    MirrorNoseParam=dict(zip(params_name,MirrorNoseParam))
    MirrorNoseParam["useMirror"]=True
    MirrorNoseParam["useMulti"]=False
    MirrorNoseParam["renderMode"]='powder'
    make["Face"]["Params"]["MirrorNose"]=MirrorNoseParam

    MirrorShadingParams=[]
    MirrorShadingParams.append([[0.1],[0.5],[0.3],[0.9],[0.2],[0.8],[0.3],[0.2,0.2,0.2],[0.91]])
    MirrorShadingParams.append([[0.3],[0.9],[0.6],[0.9],[0.5],[0.95],[0.3],[0.2,0.2,0.2],[0.91]])
    MirrorShadingParams.append([[0.6],[0.8],[0.8],[0.7],[0.7],[0.8],[0.3],[0.2,0.2,0.2],[0.91]])
    MirrorShadingParam=np.array(MirrorShadingParams).T.tolist()
    MirrorShadingParam=dict(zip(params_name,MirrorShadingParam))
    MirrorShadingParam["useMirror"]=True
    MirrorShadingParam["useMulti"]=False
    MirrorShadingParam["renderMode"]='powder'
    make["Face"]["Params"]["MirrorShading"]=MirrorShadingParam

    ShadingParams=[]
    ShadingParams.append([[0.8],[0.6],[0.8],[1-0.6],[0.9],[0.5],[0.3],[0.2,0.2,0.2],[0.91]])
    ShadingParam=np.array(ShadingParams).T.tolist()
    ShadingParam=dict(zip(params_name,ShadingParam))
    ShadingParam["useMirror"]=False
    ShadingParam["useMulti"]=False
    ShadingParam["renderMode"]='powder'
    make["Face"]["Params"]["Shading"]=ShadingParam

    MirrorHighLightParams=[]
    MirrorHighLightParams=[]
    MirrorHighLightParams.append([[0.5],[0.1],[0.5],[0.2],[0.5],[0.15],[0.2],[0.7,0.7,0.7],[0.91]])   #ほっぺ
    MirrorHighLightParams.append([[0.38],[0.4],[0.4],[0.38],[0.4],[0.4],[0.2],[0.7,0.7,0.7],[0.91]])  #目頭
    MirrorHighLightParam=np.array(MirrorHighLightParams).T.tolist()
    MirrorHighLightParam=dict(zip(params_name,MirrorHighLightParam))
    MirrorHighLightParam["useMirror"]=True
    MirrorHighLightParam["useMulti"]=False
    MirrorHighLightParam["renderMode"]='powder'
    make["Face"]["Params"]["MirrorHighLight"]=MirrorHighLightParam



    HighLightParams=[]
    HighLightParams.append([[0.7],[0.5],[0.75],[0.5],[0.725],[0.5],[0.2],[0.7,0.7,0.7],[0.91]])    
    HighLightParams.append([[0.9],[0.5],[0.95],[0.5],[0.925],[0.5],[0.2],[0.7,0.7,0.7],[0.91]])     
    HighLightParams.append([[0.1],[0.2],[0.1],[0.8],[0.1],[0.5],[0.2],[0.7,0.7,0.7],[0.91]])        
    HighLightParams.append([[0.4],[0.5],[0.5],[0.5],[0.45],[0.5],[0.2],[0.7,0.7,0.7],[0.91]])
    HighLightParams.append([[0.5],[0.5],[0.6],[0.5],[0.55],[0.5],[0.2],[0.7,0.7,0.7],[0.91]])
    HighLightParam=np.array(HighLightParams).T.tolist()
    HighLightParam=dict(zip(params_name,HighLightParam))
    HighLightParam["useMirror"]=False
    HighLightParam["useMulti"]=False
    HighLightParam["renderMode"]='powder'
    make["Face"]["Params"]["HighLight"]=HighLightParam
    


    MirrorMultiShadingParams=[]
    MirrorMultiShadingParams.append([[0.1],[0.5],[0.3],[0.9],[0.2],[0.8],[0.3],[0.2,0.2,0.2],[0.91]])
    MirrorMultiShadingParams.append([[0.3],[0.9],[0.6],[0.9],[0.5],[0.95],[0.3],[0.2,0.2,0.2],[0.91]])
    MirrorMultiShadingParams.append([[0.6],[0.8],[0.8],[0.7],[0.7],[0.8],[0.3],[0.2,0.0,0.2],[0.91]])
    MirrorMultiShadingParam=np.array(MirrorMultiShadingParams).T.tolist()
    MirrorMultiShadingParam=dict(zip(params_name,MirrorMultiShadingParam))
    MirrorMultiShadingParam["useMirror"]=True
    MirrorMultiShadingParam["useMulti"]=True
    MirrorMultiShadingParam["renderMode"]='powder'
    make["Face"]["Params"]["MirrorMultiShading"]=MirrorMultiShadingParam

