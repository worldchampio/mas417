import addons
import time
import cv2 as cv

"""
This app mainly work on Windows, other operative system may become problematic due to cross incompatebility with OpenCV
"""

def main():
    
    print("Welcome to a simple map stl file constructor")
    time.sleep(1)
    location = input("Please type in the desiered location to re-create:\n")
    area_obj = addons.AreaSelector(location)
    area_obj.position_selector_info_list()
    while True:
            specific_name = input("Is the name specific? (NB! Case-sensitive)\nOptions: yes no\nSelect(Y/N):\n")
            if specific_name == "Y" or specific_name == "N":
                break
            else:
                print("wrong input, try again")
    if specific_name == "Y":
        area_obj.specific_name_sorter()
    area_obj.position_selector()
    while True:
            area_obj.map_previewer()
            cv.waitKey(0)
            cv.destroyAllWindows()
            task_selector = input("Is the location right?\nOptions: zoom-out yes no\n Select the right option(zoom/yes/no):\n")
            if task_selector == "yes":
                break
            elif task_selector == "no":
                print("Please choose another location")
                area_obj.position_selector()
                continue
            elif task_selector == "zoom":
                area_obj.bbx_scale_pre_map_factor = int(input("Please choose the percentage to zoom out(+) (whole positive numbers only!):\n"))
                continue
            else:
                print("Wrong input, try again")
                continue
    
    print("Position selected, starting STL creator")
    time.sleep(1)
    
    stl_obj = addons.CreateSTL(area_obj.position_coordinates)

    while True:
        stl_obj.print_area_prviewer()
        cv.waitKey(0)
        cv.destroyAllWindows()
        task_selector = input("Is the bounding box size right?\nOptions: zoom-in/out yes\n Select the right option(zoom/yes):\n")
        if task_selector == "yes":
            break
        elif task_selector == "zoom":
            stl_obj.bbx_scale_factor = int(input("Please choose the percentage to zoom (Out:+ In:-) (whole numbers only!):\n"))
            continue
        else:
            print("Wrong input, try again")
            continue

    stl_obj.img_from_wms()
    stl_obj.numpy2stl()
    


    # center_E = area_obj.coord_out_E
    # center_N = area_obj.coord_out_N

    # size = area_obj.size_out/50 #Size stjålet fra AreaSelector klass
    # y = 1

    # bbox_arr = np.array([center_E-(size*y),center_N-(size*y), \
    #         center_E+(size*y),center_N+(size*y)])
    # bbox_in = str(bbox_arr[0])+','+str(bbox_arr[1])+','+str(bbox_arr[2])+','+str(bbox_arr[3])

    # # Instantiate STL class
    # obj = addons.CreateSTL()

    # # Get test image from geonorge WMS
    # image_wms = obj.img_from_wms(bbox_in)

    # # Convert from img -> np -> stl
    # obj.numpy2stl(image_wms,'fjell.stl',solid=True,scale=0.2)

if __name__ == "__main__":
    main()