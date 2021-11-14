
import numpy as np
import requests
import json
import pandas as pd
import os
import openpyxl
from PIL import Image
from io import BytesIO
import shutil
import sys
import cv2 as cv
import keyboard
import time

class AreaSelector():

    def __init__(self, position_name) -> None:
        self.position_name = position_name
        self.position_coordinates = np.array([0,0])
        self.position_bbox = np.array([0,0,0,0])
        self.bbx_scale_pre_map = 5000
        self.bbx_scaling_factor = 1 #Default scaling
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
            'TRANSPARENT=True&' 
            #Width and height used for a normal 1080p monitor, which is assumed to be the standard



    def map_previewer(self):
        #Selecting the right coordinates
        coordinates = self.df_places['koordinater(E/N)'][self.selected_location]

        #Converting from WGS84 to EPSG:25833
        factor_east = 17082.7258123
        factor_north = 111108.084509015

        east = (coordinates[0])*factor_east
        north = (coordinates[1])*factor_north

        #Adjusting self.bbx_scaling_factor if want larger map preview
        y = self.bbx_scale_pre_map
        bbx_pre = np.array([east-5000,north-5000, \
            east+5000,north+5000])


        resquest_url = f'{self.map_url}&BBOX={bbx_pre[0]},{bbx_pre[1]},{bbx_pre[2]},{bbx_pre[3]}'
        response = requests.get(resquest_url, verify=True)  # SSL Cert verification explicitly enabled. (This is also default.)
        # print(f"HTTP response status code = {response.status_code}")
        if response.status_code != 200:
            raise RuntimeError('Invalid map area or no connection with geonorge.no')
        # print(type(response.content))
        # print(sys.getsizeof(response.content))

        img = Image.open(BytesIO(response.content))

        img.save('temp_map_area_pre.png')  

        img = cv.imread(cv.samples.findFile("temp_map_area_pre.png"))
        height, width = img.shape[:2]

        #Applying some pointer circles to highlight the selected area
        cv.circle(img,(int(width/2),int(height/2)), 2, (0,0,255), 2)
        cv.circle(img,(int(width/2),int(height/2)), 50, (0,0,255), 2)
        cv.circle(img,(int(width/2),int(height/2)), 100, (0,0,255), 2)

        
        print("Displaying preview of selected position, hit ESC to enter")
        time.sleep(2)
        cv.imshow("Preview Position", img)
        cv.waitKey(0)



    def position_selector(self):
        if len(self.df_places) > 10:
            excel_file_path = f'{os.getcwd()}/place_list_browser.xlsx'
            print(f"The location searched for yielded more than 10 results, results are listed in {excel_file_path} for further inspection")
            self.df_places.to_excel(excel_file_path)
        else:
            print(self.df_places)

        while True:
            selected_location = input("Please browse for the correct position and enter the index number\n")
            try:
                self.selected_location = int(selected_location)
                break
            except ValueError:
                print("Input is not an integer, try again")
                continue

                
        


        

    



    def position_selector_info_list(self):
        search_url = f'{self.position_url}sok={self.position_name}&treffPerSide=500&side=1'
        response = requests.get(search_url, verify=True)

        response_dict = response.json()
        response_places = response_dict['navn']

        if not response_places:
            raise NameError('No result')

        # print(response_places[0])
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
        print(self.df_places)

    def specific_name_sorter(self):
        sorter = self.df_places['skrivemåte']
        for col in range(len(self.df_places)):
            
            if sorter[col] != self.position_name:
                self.df_places.drop([col], inplace=True)
    
        self.df_places.reset_index(drop=True, inplace=True)
        # print(self.df_places)





if __name__ == "__main__":
    pass


    

