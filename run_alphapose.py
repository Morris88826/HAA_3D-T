import os
import glob

if __name__ == "__main__":

    input_sources = glob.glob('./dataset/raw/belly_dancing/belly_dancing_000')


    for input_source in input_sources:
        alphapose_outpath = './dataset/alphapose'
        if not os.path.exists(alphapose_outpath):
            os.mkdir(alphapose_outpath)

        class_name = input_source.split('/')[-2]
        video_name = input_source.split('/')[-1]

        if not os.path.exists(alphapose_outpath+'/'+class_name):
            os.mkdir(alphapose_outpath+'/'+class_name)

        out_path = alphapose_outpath+ '/' + class_name + '/' + video_name + '.json'

        os.system('python ./AlphaPose/scripts/demo_inference.py --cfg ./AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint ./AlphaPose/pretrained_models/fast_res50_256x192.pth --indir {} \
                --outdir {} --out_name {}'.format(input_source, alphapose_outpath+'/'+class_name, out_path))
