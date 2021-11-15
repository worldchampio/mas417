import numpy as np
import requests
import json
import pandas as pd
import os
import platform
import openpyxl
from PIL import Image
from io import BytesIO
import shutil
import sys
#import cv2 as cv        <- cv is imported below (on Ubuntu)
import time
import struct
from itertools import product
try:
    from .cwrapped import tessellate
    c_lib = True
except ImportError:
    c_lib = False


class CreateSTL():

    def __init__(self) -> None:
        self.ASCII_FACET = """  facet normal  {face[0]:e}  {face[1]:e}  {face[2]:e}
              outer loop
                vertex    {face[3]:e}  {face[4]:e}  {face[5]:e}
                vertex    {face[6]:e}  {face[7]:e}  {face[8]:e}
                vertex    {face[9]:e}  {face[10]:e}  {face[11]:e}
              endloop
            endfacet"""

        self.BINARY_HEADER = "80sI"
        self.BINARY_FACET = "12fH"

    def _build_binary_stl(self,facets):
        """returns a string of binary binary data for the stl file"""
        lines = [struct.pack(self.BINARY_HEADER, b'Binary STL Writer', len(facets)), ]
        for facet in facets:
            facet = list(facet)
            facet.append(0)  # need to pad the end with a unsigned short byte
            lines.append(struct.pack(self.BINARY_FACET, *facet))
        return lines

    def _build_ascii_stl(self,facets):
        """returns a list of ascii lines for the stl file """
        lines = ['solid ffd_geom', ]
        for facet in facets:
            lines.append(self.ASCII_FACET.format(face=facet))
        lines.append('endsolid ffd_geom')
        return lines

    def writeSTL(self, facets, file_name, ascii=False):
        """writes an ASCII or binary STL file"""

        f = open(file_name, 'wb')
        if ascii:
            lines = self._build_ascii_stl(facets)
            lines_ = "\n".join(lines).encode("UTF-8")
            f.write(lines_)
        else:
            data = self._build_binary_stl(facets)
            data = b"".join(data)
            f.write(data)

        f.close()

    def roll2d(self, image, shifts):
        return np.roll(np.roll(image, shifts[0], axis=0), shifts[1], axis=1)

    def numpy2stl(self, A, fn, scale=0.1, mask_val=None, ascii=False,
                max_width=100.,
                max_depth=60.,
                max_height=30.,
                solid=False,
                min_thickness_percent=0.1,
                force_python=False):
    
        m, n = A.shape
        #m = size_x
        #n = size_y

        if n >= m:
            # rotate to best fit a printing platform
            A = np.rot90(A, k=3)
            m, n = n, m
        A = scale * (A - A.min())

        if not mask_val:
            mask_val = A.min() - 1.

        if c_lib and not force_python:  # try to use c library
            # needed for memoryviews
            A = np.ascontiguousarray(A, dtype=float)

            facets = np.asarray(tessellate(A, mask_val, min_thickness_percent,
                                solid))
            # center on platform
            facets[:, 3::3] += -m / 2
            facets[:, 4::3] += -n / 2

        else:  # use python + numpy
            facets = []
            mask = np.zeros((m, n))
            print("Lager 3D modell...")
            for i, k in product(range(m - 1), range(n - 1)):

                this_pt = np.array([i - m / 2., k - n / 2., A[i, k]])
                top_right = np.array([i - m / 2., k + 1 - n / 2., A[i, k + 1]])
                bottom_left = np.array([i + 1. - m / 2., k - n / 2., A[i + 1, k]])
                bottom_right = np.array(
                    [i + 1. - m / 2., k + 1 - n / 2., A[i + 1, k + 1]])

                n1, n2 = np.zeros(3), np.zeros(3)

                if (this_pt[-1] > mask_val and top_right[-1] > mask_val and
                        bottom_left[-1] > mask_val):

                    facet = np.concatenate([n1, top_right, this_pt, bottom_right])
                    mask[i, k] = 1
                    mask[i, k + 1] = 1
                    mask[i + 1, k] = 1
                    facets.append(facet)

                if (this_pt[-1] > mask_val and bottom_right[-1] > mask_val and
                        bottom_left[-1] > mask_val):

                    facet = np.concatenate(
                        [n2, bottom_right, this_pt, bottom_left])
                    facets.append(facet)
                    mask[i, k] = 1
                    mask[i + 1, k + 1] = 1
                    mask[i + 1, k] = 1
            facets = np.array(facets)

            if solid:
                #print("Computed edges...")
                edge_mask = np.sum([self.roll2d(mask, (i, k))
                                for i, k in product([-1, 0, 1], repeat=2)],
                                axis=0)
                edge_mask[np.where(edge_mask == 9.)] = 0.
                edge_mask[np.where(edge_mask != 0.)] = 1.
                edge_mask[0::m - 1, :] = 1.
                edge_mask[:, 0::n - 1] = 1.
                X, Y = np.where(edge_mask == 1.)
                locs = zip(X - m / 2., Y - n / 2.)

                zvals = facets[:, 5::3]
                zmin, zthickness = zvals.min(), zvals.ptp()

                minval = zmin - min_thickness_percent * zthickness

                bottom = []
                #print("Extending edges, creating bottom...")
                for i, facet in enumerate(facets):
                    if (facet[3], facet[4]) in locs:
                        facets[i][5] = minval
                    if (facet[6], facet[7]) in locs:
                        facets[i][8] = minval
                    if (facet[9], facet[10]) in locs:
                        facets[i][11] = minval
                    this_bottom = np.concatenate(
                        [facet[:3], facet[6:8], [minval], facet[3:5], [minval],
                        facet[9:11], [minval]])
                    bottom.append(this_bottom)

                facets = np.concatenate([facets, bottom])

        xsize = facets[:, 3::3].ptp()
        if xsize > max_width:
            facets = facets * float(max_width) / xsize

        ysize = facets[:, 4::3].ptp()
        if ysize > max_depth:
            facets = facets * float(max_depth) / ysize

        zsize = facets[:, 5::3].ptp()
        if zsize > max_height:
            facets = facets * float(max_height) / zsize

        self.writeSTL(facets, fn, ascii=ascii)

    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    def img_from_wms(self, bbox_input):

        #Request body
        request_url = 'https://wms.geonorge.no/skwms1/wms.hoyde-dom?' \
           'SERVICE=WMS&' \
           'VERSION=1.3.0&' \
           'REQUEST=GetMap&' \
           'FORMAT=image/png&' \
           'TRANSPARENT=false&' \
           'LAYERS=DOM:None&' \
           'CRS=EPSG:25833&' \
           'STYLES=&' \
           'WIDTH=1751&' \
           'HEIGHT=1241&' \
           'BBOX='+str(bbox_input)

        response = requests.get(request_url, verify=True)  # SSL Cert verification explicitly enabled. (This is also default.)
        #print(f"HTTP response status code = {response.status_code}")
        img = Image.open(BytesIO(response.content))
    
        #Convert to array and grayscale
        np_img = np.asarray(img)
        img_gray = self.rgb2gray(np_img)
        return img_gray

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
        """
        img = cv.imread(cv.samples.findFile("temp_map_area_pre.png"))
        height, width = img.shape[:2]
        
        #Applying some pointer circles to highlight the selected area
        cv.circle(img,(int(width/2),int(height/2)), 2, (0,0,255), 2)
        cv.circle(img,(int(width/2),int(height/2)), 50, (0,0,255), 2)
        cv.circle(img,(int(width/2),int(height/2)), 100, (0,0,255), 2)
        
        #Applying some text to indicate how to exit the window
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, "Hit ESC to EXIT", (int(width/2)-120,int(height/2)-125), font, 1,(0,0,255), 2, cv.LINE_AA )
        """
        print("Displaying preview of selected position, hit ESC to exit")
        time.sleep(4)
        #cv.imshow("Preview Position. Hit ESC to EXIT", img)

        #Setting the new coordinates to the instance variable to be used in the STL converter
        self.coord_out_E = east
        self.coord_out_N = north
        self.size_out = size



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