from flask import Flask, request, jsonify
import threading
import cv2
from PIL import Image
import torchvision.transforms as standard_transforms
from collections import OrderedDict
import torch
from model.locator import Crowd_locator
from misc.utils import *
import numpy as np

app = Flask(__name__)

# Dictionary to store results
results = {}

def setup_model():
    GPU_ID = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    torch.backends.cudnn.benchmark = True
    netName = 'HR_Net'
    model_path = '../PretrainedModels/SHHA-HR-ep_905_F1_0.715_Pre_0.760_Rec_0.675_mae_112.3_mse_229.9.pth'
    net = Crowd_locator(netName, GPU_ID, pretrained=True)
    net.cuda()
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    return net

def img_transform():
    mean_std = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])
    return standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

def process_stream(rtsp_url, net):
    cap = cv2.VideoCapture(rtsp_url)
    transform = img_transform()
    if not cap.isOpened():
        print(f"Failed to open stream: {rtsp_url}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Process the frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_transformed = transform(img).unsqueeze(0).cuda()

        with torch.no_grad():
            pred_threshold, pred_map, _ = [i.cpu() for i in net(img_transformed, mask_gt=None, mode='val')]
            binar_map = torch.where(pred_map >= pred_threshold, torch.ones_like(pred_map), torch.zeros_like(pred_map))

        # Extract and update results
        pred_data, _ = get_boxInfo_from_Binar_map(binar_map.numpy())
        results[rtsp_url] = {
            'num_objects': pred_data['num'],
            'objects': [{'x': int(point[0]), 'y': int(point[1])} for point in pred_data['points']]
        }

def get_boxInfo_from_Binar_map(Binar_numpy, min_area=3):
    Binar_numpy = Binar_numpy.squeeze().astype(np.uint8)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(Binar_numpy, connectivity=4)
    boxes = stats[1:, :]
    points = centroids[1:, :]
    index = (boxes[:, 4] >= min_area)
    boxes = boxes[index]
    points = points[index]
    return {'num': len(points), 'points': points}, boxes

@app.route('/analyze', methods=['GET'])
def analyze_stream():
    rtsp_url = request.args.get('rtsp_url')
    if rtsp_url not in results:
        return jsonify({'error': 'No data available for this URL or URL is invalid'}), 404
    return jsonify(results[rtsp_url])

if __name__ == '__main__':
    net = setup_model()
    # Define your CCTV RTSP URLs
    cctv_urls = ["rtsp://210.99.70.120:1935/live/cctv001.stream",
                 "rtsp://210.99.70.120:1935/live/cctv002.stream"]
    for url in cctv_urls:
        thread = threading.Thread(target=process_stream, args=(url, net))
        thread.daemon = True
        thread.start()
    app.run(debug=True, port=5000)
