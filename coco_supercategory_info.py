import os
import json


def main():
    for s in ["train", "val"]:
        with open("data/coco/annotations/panoptic_{}2017.json".format(s), "r") as COCO:
            js = json.loads(COCO.read())
            superclasses = []
            dict = {}
            """
            background = {}
            superclasses.append(0)
            background["supercategory"] = "background"
            background["super_id"] = 0
            background["coco_label"] = "background"
    
            dict[str(0)] = background
            """

            #print(json.dumps(js["categories"]))
            for entry in js["categories"]:
                sc_info = {}
                sc = entry["supercategory"]
                label = entry["id"]
                if sc not in superclasses:
                    superclasses.append(sc)

                if label not in dict:
                    sc_info["supercategory"] = sc
                    sc_info["isthing"] = entry["isthing"]
                    sc_info["super_id"] = superclasses.index(sc) + 1
                    sc_info["coco_label"] = entry["name"]
                    dict[label] = sc_info

            with open(os.path.join("other", "coco_{}_supercategories_map_by_id.json".format(s)), 'w') as d:
                json.dump(dict, d)

if __name__ == "__main__":
    main()
