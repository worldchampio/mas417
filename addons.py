
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

class AreaSelector():

    def __init__(self, position_name) -> None:
        self.position_name = position_name
        self.position_coordinates = np.array([0,0])
        self.position_bound_box = np.array([0,0,0,0])
        self.scaling_factor = 1 #Default scaling
        #EXAMPLE https://ws.geonorge.no/stedsnavn/v1/navn?sok=Dalen&treffPerSide=100&side=1 
        self.position_url = 'https://ws.geonorge.no/stedsnavn/v1/navn?'
        self.map_url = 'https://openwms.statkart.no/skwms1/wms.topo4?' \
            'SERVICE=WMS&' \
            'VERSION=1.3.0&' \
            'REQUEST=GetMap&' \
            'FORMAT=image/png&' \
            'TRANSPARENT=True&' \
            'CRS=EPSG:25833&' \
            'LAYERS=topo4_WMS&' \
            'SIZE=(1920,1080)&' \
            'BBOX=18676.05018252586,6845773.541229122,117163.78576648101,6915575.52858474'





    def map_previewer(self):
        response = requests.get(self.map_url, verify=True)  # SSL Cert verification explicitly enabled. (This is also default.)
        print(f"HTTP response status code = {response.status_code}")
        print(type(response.content))
        print(sys.getsizeof(response.content))

        img = Image.open(BytesIO(response.content))

        img.save('tester2.png')  

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
        df_places['koordinater(Ø/N)'] = response_coordinate
        self.df_places = df_places
        # df_places.to_excel(f'{os.getcwd()}/tester.xlsx')

    def specific_name_sorter(self):
        sorter = self.df_places['skrivemåte']
        for col in range(len(self.df_places)):
            
            if sorter[col] != self.position_name:
                self.df_places.drop([col], inplace=True)
    
        self.df_places.reset_index(drop=True, inplace=True)
        print(self.df_places)





if __name__ == "__main__":
    test = AreaSelector("Dalen")
    # test.position_selector_info_list()
    # test.specific_name_sorter()
    # test.df_places.to_excel(f'{os.getcwd()}/tester.xlsx')
    test.map_previewer()


    

