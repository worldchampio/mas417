import numpy as np
import requests
import pandas as pd
import os
import cv2 as cv
import time
import shlex
import sys
from PIL import Image
from io import BytesIO
from sys import platform


class AreaSelector():

    def __init__(self, position_name) -> None:
        self.position_name = position_name
        self.position_coordinates = np.array([0,0])
        self.bbx_scale_pre_map_factor = 0
        self.pre_map_default_size = 40000
        self.position_selector_flag = False
        #EXAMPLE https://ws.geonorge.no/stedsnavn/v1/navn?sok=Dalen&treffPerSide=100&side=1 
        self.position_url = 'https://ws.geonorge.no/stedsnavn/v1/navn?'
        self.map_url = 'https://openwms.statkart.no/skwms1/wms.topo4?' \
            'SERVICE=WMS&' \
            'VERSION=1.3.0&' \
            'REQUEST=GetMap&' \
            'FORMAT=image/png&' \
            'STYLES=&' \
            'CRS=EPSG:25833&' \
            'LAYERS=topo4_WMS&' \
            'WIDTH=1050&' \
            'HEIGHT=1050&' \
            'TRANSPARENT=True&' \
            #Width and height used for a normal 1080p monitor, which is assumed to be the standard




    def map_previewer(self) -> None:
        """Function which lets the user preview the selected location before constructing the STL"""

        #Selecting the right coordinates
        coordinates = self.df_places['koordinater(E/N)'][self.selected_location]

        east = float(coordinates[0])
        north = float(coordinates[1])

        #Adjusting self.bbx_scaling_factor if user want larger map preview
        #Making variables for instance.variables in order to keep boundingbox equation cleaner
        y = 1+(self.bbx_scale_pre_map_factor/100)
        size = self.pre_map_default_size
        bbx_pre = np.array([east-(size*y),north-(size*y), \
            east+(size*y),north+(size*y)])

        resquest_url = f'{self.map_url}BBOX={bbx_pre[0]},{bbx_pre[1]},{bbx_pre[2]},{bbx_pre[3]}'
        response = requests.get(resquest_url, verify=True)  # SSL Cert verification explicitly enabled. (This is also default.)
        # print(f"HTTP response status code = {response.status_code}")
        if response.status_code != 200:
            raise RuntimeError('Invalid map area or no connection with geonorge.no')

        img = Image.open(BytesIO(response.content))

        img.save('temp_map_area_pre.png')  

        img = cv.imread(cv.samples.findFile("temp_map_area_pre.png"))
        height, width = img.shape[:2]

        #Applying some pointer circles to highlight the selected area
        cv.circle(img,(int(width/2),int(height/2)), 2, (0,0,255), 2)
        cv.circle(img,(int(width/2),int(height/2)), 50, (0,0,255), 2)
        cv.circle(img,(int(width/2),int(height/2)), 100, (0,0,255), 2)

        #Applying some text to indicate how to exit the window
        # cv.putText(img, "Hit ESC to EXIT", (int(width/2),int(height/2)+200), fontFace=cv.FONT_HERSHEY_SIMPLEX, 4 ,(0,0,255), )
        
        print("Displaying preview of selected position, hit ESC to exit")
        time.sleep(4)
        cv.imshow("Preview Position. Hit ESC to EXIT", img)




    def position_selector(self) -> None:
        """Lets the user choose the correct place from the results generated in position_selector_info_list"""


        #Checks if the results are more than 10 and in that case lists them in an excel file to not drown the terminal
        if len(self.df_places) > 10:
            excel_file_path = f'{os.getcwd()}/place_list_browser.xlsx'
            print(f"The location searched for yielded more than 10 results, results are listed in {excel_file_path} for further inspection")
            
            #Checking if the user has an open excel window, which wil block the program from writing a new list
            if self.position_selector_flag == False and os.path.isfile(excel_file_path):
                wait = input("Please make sure the place_list_browser.xlsx file is closed to avoid blocking generating of new list\nThen hit ENTER")

            #Checking if the user has the excel file already open to avoid an IO error
            try:
                self.df_places.to_excel(excel_file_path)
                #Checking operating system to select correct method for automatic file opening
                if platform == "linux" or platform == "darwin":
                    os.system("open " + shlex.quote(excel_file_path))
                else:
                    os.system("start " + excel_file_path)
            except IOError:
                if self.position_selector_flag == False:
                    print("No new list file were constructed, please close current place_list_browser.xlsx file and restart the program")
                    sys.exit()
        else:
            print(self.df_places)

        #Stating that the program has now ran its first time
        self.position_selector_flag = True

        #The user selects their place by typing in the index number
        while True:
            selected_location = input("Please browse for the correct position and enter the index number\n")
            try:
                self.selected_location = int(selected_location)
                break
            except ValueError:
                print("Input is not an integer, try again")
                continue



    def position_selector_info_list(self) -> None:
        """A function gathering all the results after searching for a specific place. NB! Maximum results is 500"""

        search_url = f'{self.position_url}sok={self.position_name}&treffPerSide=500&side=1&utkoordsys=25833'
        response = requests.get(search_url, verify=True)

        response_dict = response.json()
        response_places = response_dict['navn']

        #The app is shut down if no results are found and user has to restart it
        if not response_places:
            raise NameError('No result')

        df_places = pd.DataFrame.from_dict(response_places)

        ### Unpacking nested dicts for kommuner and fylker and generating independent coordinate row
        response_counties = []
        response_municipality = []
        response_coordinate = []
        for col in range(len(response_places)):
            response_municipality.append(response_places[col]['kommuner'][0]['kommunenavn'].split()[0])
            response_counties.append(response_places[col]['fylker'][0]['fylkesnavn'].split()[0])
            response_coordinate.append([response_places[col]['representasjonspunkt']['øst'], response_places[col]['representasjonspunkt']['nord']])

        df_places.drop(['kommuner', 'fylker', 'språk', 'navnestatus', 'representasjonspunkt', 'stedsnummer', 'skrivemåtestatus', 'stedstatus'], axis='columns', inplace=True)
        df_places['kommune'] = response_municipality
        df_places['fylke'] = response_counties
        df_places['koordinater(E/N)'] = response_coordinate
        self.df_places = df_places




    def specific_name_sorter(self) -> None:
        """ A Function used to only list name specific places. Example: Searching Dalen and selescting this function will
            make results like Hasseldalen dissapear"""

        sorter = self.df_places['skrivemåte']
        for col in range(len(self.df_places)):
            
            if sorter[col] != self.position_name:
                self.df_places.drop([col], inplace=True)
    
        self.df_places.reset_index(drop=True, inplace=True)


if __name__ == "__main__":
    pass