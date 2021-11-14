import addons as add

if __name__ == "__main__":
    
    example_coords = '18676.05018252586,6845773.541229122,117163.78576648101,6915575.52858474' #Jostedalsbreen?
    
    # Instantiate STL class
    obj = add.CreateSTL()

    # Get bounding box from geonorge topo
    coords = obj.create_bbox((18676.05018252586,6845773.541229122))

    # Get test image from geonorge WMS
    image_wms = obj.img_from_wms(coords)

    # Convert from img -> np -> stl
    obj.numpy2stl(image_wms,'fjell.stl',solid=True)

    