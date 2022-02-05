import Adadetectron
import AdabinsFeatures

import torchvision.transforms.functional as TTF
from torchvision.transforms import InterpolationMode

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
# from custom_train import train
from AdabinsOriginal.models.miniViT import mViT
from AdabinsOriginal.models.unet_adaptive_bins import Encoder
from AdabinsOriginal.models.unet_adaptive_bins import DecoderBN


def main():
    """
    encoder, decoder = AdabinsFeatures.extract_adabins_feature_encoder_decoder()
    nyu_sample = "data/NYUv2-raw/raw/sync/bathroom_0030/rgb_00000.jpg"

    panoptic_predictor = Adadetectron.get_panoptic_predictor()
    panoptic_output_ids, segmentation_info = Adadetectron.get_panoptic_info()
    """


def convert_array_to_listdict(x, target_size):
    list = []
    for img in x:
        #detectron2 expects BGR, not RGB -> or set cfg.INPUT.FORMAT in Adadetectron.py
        #img = convert_rbg_to_bgr(img)
        #originally a float tensor range(0,1) but detectron expects range(0,255) -> change dataloader
        dict = {'image': TTF.resize(img, size=target_size, interpolation=InterpolationMode.BICUBIC)} #for tensors only BICUBIC and NEAREST supported for backpropagation
        #print(dict['image'].shape)
        list.append(dict)
        """
        cv2.imshow("debug", mat=dict['image'].cpu().numpy().transpose(1,2,0))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        """

    return list

def convert_rbg_to_bgr(x):
    #this is wrong
    bgr = [2, 1, 0]
    x = x[:, bgr]
    return x


class PanopticAdabinsCombined(nn.Module):
    # based on Adabins
    def __init__(self, backend, n_bins=100, min_val=0.1, max_val=10, norm='linear'):
        super(PanopticAdabinsCombined, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        #we load a pretrained AdaBins model here and extract the encode-decoder part, for now we use NYUv2 instead of KITTI
        self.encoder, self.decoder = AdabinsFeatures.extract_adabins_feature_encoder_decoder()
        #self.encoder = Encoder(backend)
        #self.decoder = DecoderBN(num_classes=128)

        self.panoptic_predictor = Adadetectron.get_panoptic_custom_model()#Adadetectron.get_panoptic_predictor()

        #we need to increase input dimension by 1 since we added a new panoptic layer to the feature map
        self.adaptive_bins_layer = mViT(129, n_query_channels=129, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=129, norm=norm, num_heads=3)
        #again input dimension increases by 1
        self.conv_out = nn.Sequential(nn.Conv2d(129, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))

    def forward(self, x, **kwargs):
        input_adabins_raw = x[0]
        input_detectron_raw = x[1]#x[0] * 255
        unet_out = self.decoder(self.encoder(input_adabins_raw), **kwargs)

        feature_height = unet_out.size(dim=2)
        feature_width = unet_out.size(dim=3)
        feature_image_dim = [feature_height, feature_width]

        x_converted = convert_array_to_listdict(input_detectron_raw / 255.0, feature_image_dim)

        batch_detectron_output = self.panoptic_predictor(x_converted)

        """
        extract panoptic map from detectron output
        we are only interested in the map here, the info that describes the meaning of these ids is not used for now
        maybe we need to include it later for visualization
        """
        panoptics = []
        #i = 0
        for output in batch_detectron_output:
            map, _ = output["panoptic_seg"]
            #match feature map dimensions -> for now in handled before passing to detectron to reduce strain, but detectrons performance won't be optimal
            """"
            map_resized = F.interpolate(map, feature_image_dim).clamp(min=0, max=255)#TTF.resize(map, size=feature_image_dim, interpolation=InterpolationMode.BICUBIC)
            
            cv2.imshow("x_converted", mat=x_converted[i]['image'].cpu().numpy().transpose(1, 2, 0))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

            #Adadetectron.visualize(input_detectron_raw[i], map, info)
            panoptics.append(map)
            #i+=1

        # Concatenate feature map with panoptic map-> hxwx(#features+1)
        panoptic_adabins_features = torch.zeros((unet_out.size(dim=0), unet_out.size(dim=1)+1, feature_height, feature_width), device='cuda')

        #iterate batches
        for i in range(len(unet_out)):
            panoptics[i] = torch.reshape(panoptics[i], (1,feature_height,feature_width))
            panoptic_adabins_features[i] = torch.cat((unet_out[i], panoptics[i]), 0)

        #print("concatenated: "+str(panoptic_adabins_features.shape))

        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(panoptic_adabins_features)
        out = self.conv_out(range_attention_maps)

        # Post process
        # n, c, h, w = out.shape
        # hist = torch.sum(out.view(n, c, h * w), dim=2) / (h * w)  # not used for training

        # Equation 2 from AdaBins paper
        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        # Equation 3 from AdaBins paper, pred -> predicted depth value
        pred = torch.sum(out * centers, dim=1, keepdim=True)
        #print("PRED: " + str(pred.shape))
        #print("PRED_0: " + str(pred[0][0]))
        #print("Bind Edges: " + str(bin_edges))
        return bin_edges, pred

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, n_bins, **kwargs):
        #this step could be skipped later, I keep it for now to make the setup simpler, the basemodel is unused in the model, so this won't have any effect
        basemodel_name = 'tf_efficientnet_b5_ap'

        print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        m = cls(basemodel, n_bins=n_bins, **kwargs)
        print('Done.')
        return m


if __name__ == "__main__":
    main()
