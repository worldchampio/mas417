import addons
import time
import numpy as np

if __name__ == "__main__":
    
    print("Welcome to a simple map stl file constructor")
    time.sleep(1)
    location = input("Please type in the desiered location to re-create:\n")
    area_obj = addons.AreaSelector(location)
    area_obj.position_selector_info_list()
    while True:
            specific_name = input("Is the name specific?(Y/N):\n")
            if specific_name == "Y" or specific_name == "N":
                break
            else:
                print("wrong input, try again")
    if specific_name == "Y":
        area_obj.specific_name_sorter()
    area_obj.position_selector()
    while True:
            area_obj.map_previewer()
            #cv.waitKey(0)
            #cv.destroyAllWindows()
            task_selector = input("Is the location right?\nOptions: zoom-out yes no\n Select the right option(zoom/yes/no):\n")
            if task_selector == "yes":
                break
            elif task_selector == "no":
                print("Please choose another location")
                area_obj.position_selector()
                continue
            elif task_selector == "zoom":
                area_obj.bbx_scale_pre_map_factor = int(input("Please choose the percentage to zoom out (whole numbers only!):\n"))
                continue
            else:
                print("Wrong input, try again")
                continue
    
    center_E = area_obj.coord_out_E
    center_N = area_obj.coord_out_N

    size = area_obj.size_out/10 #Size stjålet fra AreaSelector klass
    y = 1

    bbox_arr = np.array([center_E-(size*y),center_N-(size*y), \
            center_E+(size*y),center_N+(size*y)])
    bbox_in = str(bbox_arr[0])+','+str(bbox_arr[1])+','+str(bbox_arr[2])+','+str(bbox_arr[3])

    # Instantiate STL class
    obj = addons.CreateSTL()

    # Get test image from geonorge WMS
    image_wms = obj.img_from_wms(bbox_in)

    # Convert from img -> np -> stl
    obj.numpy2stl(image_wms,'fjell.stl',solid=True,scale=0.2)