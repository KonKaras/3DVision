import os
import json
import numpy as np


def main():
    for s in ["train", "val"]:
        prepare_json(s)

def prepare_json(datasubset):
    for datatype in ["panoptic", "instances"]:
        print("Preparing " + datatype + " " + datasubset)
        with open("data/coco/annotations/{0}_{1}2017.json".format(datatype, datasubset), "r") as COCO:
            dict = {}
            js = json.loads(COCO.read())
            dict["info"] = js["info"]
            dict["licenses"] = js["licenses"]
            dict["images"] = js["images"]
            annotations = js["annotations"]

            with open("other/coco_{}_supercategories_map_by_id.json".format(datasubset), "r") as MAP:
                mapping = json.loads(MAP.read())

                for anno in annotations:
                    if datatype == "panoptic":
                        seg_info = anno["segments_info"]
                        for entry in seg_info:
                            entry["category_id"] = mapping[str(entry["category_id"])]["super_id"]
                    else:
                        anno["category_id"] = mapping[str(anno["category_id"])]["super_id"]
                dict["annotations"] = annotations
                known_supers = []
                new_categories = []
                for category in mapping:
                    supercategory = {}
                    supercategory["supercategory"] = mapping[category]["supercategory"]
                    supercategory["isthing"] = mapping[category]["isthing"]
                    if supercategory["supercategory"] not in known_supers:
                        if (datatype == "panoptic" and supercategory["isthing"] == 0) or (datatype == "instances" and supercategory["isthing"] == 1):
                            supercategory["id"] = mapping[category]["super_id"]
                            supercategory["name"] = mapping[category]["supercategory"]
                            new_categories.append(supercategory)
                            known_supers.append(supercategory["supercategory"])

                dict["categories"] = new_categories

                with open("data/coco/annotations/{0}_{1}2017_superid.json".format(datatype, datasubset), 'w') as d:
                    json.dump(dict, d)

if __name__ == "__main__":
    main()
