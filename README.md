# .STfjell app
3D Print geographical features in norway with .stl generation


*Written by Eirik Norrheim Larsen & Magnus Nilsen*


## Description
Use several APIs to fetch data based on simple user input to create a user-customizable product, in this case a 3D-print ready .stl file.

### Quick-start
- Create a Python3 virtualenv to match `dependencies.txt`
    ```console
    $ cd <sourcedirectory>        
    $ python3 -m venv env
    $ source env/bin/activate    #confirm pip is installed with 'which pip', should output path

    $ pip install <package>      #install packages listed in dependencies
    ```

- Run `app.py` and follow the prompts

- Print the resulting `.stl` file in a 3D printer of your choice

### Update dependencies
With virtualenv activated:

    ```console
    $ pip list --local > dependencies.txt
    ```

### APIs
- [Kartverket WMS, Topografi, Stedsnavn](https://wms.geonorge.no/skwms1/wms.hoyde-dom?)

### Credits
- [STL generation methods](https://github.com/RobertABT/heightmap)

### Scope
The current and planned endpoint of this project is a generated `.stl` file matching the geographical features of the location name provided by the user and fetched from live data using several APIs.