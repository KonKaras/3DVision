from models import UnetAdaptiveBins
import model_io
import cv2

MIN_DEPTH = 1e-3
MAX_DEPTH = 10

N_BINS = 256


def extract_adabins_feature_encoder_decoder():
    model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH)
    pretrained_path = "./pretrained/AdaBins_nyu.pt"
    model, _, _ = model_io.load_checkpoint(pretrained_path, model)

    encoder = model.encoder
    encoder.requires_grad = False

    decoder = model.decoder
    decoder.requires_grad = False
    return encoder, decoder


def main():
    extract_adabins_feature_encoder_decoder()


if __name__ == "__main__":
    main()
