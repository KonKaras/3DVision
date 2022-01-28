mkdir data/coco
cd data/coco
mkdir images
cd images

#wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
#wget -c http://images.cocodataset.org/zips/test2017.zip
#wget -c http://images.cocodataset.org/zips/unlabeled2017.zip

#unzip train2017.zip
unzip val2017.zip
#unzip test2017.zip
#unzip unlabeled2017.zip

#rm train2017.zip
rm val2017.zip
#rm test2017.zip
#rm unlabeled2017.zip

cd ../
wget -c http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
#wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
#wget -c http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
#wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip
#wget -c http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

unzip panoptic_annotations_trainval2017.zip
unzip annotations/panoptic_train2017.zip
unzip annotations/panoptic_val2017.zip
#unzip annotations_trainval2017.zip
#unzip stuff_annotations_trainval2017.zip
#unzip image_info_test2017.zip
#unzip image_info_unlabeled2017.zip

rm panoptic_annotations_trainval2017.zip
rm annotations/panoptic_train2017.zip
rm annotations/panoptic_val2017.zip
#rm annotations_trainval2017.zip
#rm stuff_annotations_trainval2017.zip
#rm image_info_test17.zip
#rm image_info_unlabeled2017.zip20