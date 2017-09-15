import json
import sys

from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

from math import sqrt
from numpy import linalg
from numpy import array

#initialize clarifai and get face embedding model
app = ClarifaiApp()
model = app.models.get("d02b4508df58432fbb84e800597b8959")

#dataset
photo = sys.argv[1]
my_image = ClImage(file_obj=open(photo,'rb'))

#function to get embedding from image
def getEmbedding(image):

    #call face embedding model
    jsonTags= model.predict([image])

    #store vectors in a given photo
    faceEmbed = []

    '''iterate through every person and store
    each face embedding in array'''

    for faces in jsonTags['outputs'][0]['data']['regions']:
        for face in faces['data']['embeddings']:
            embeddingVector = face['vector']
            faceEmbed.append(embeddingVector)
    return faceEmbed

#get embeddings from the image
faces = getEmbedding(my_image)

face1 = array(faces[0])
face2 = array(faces[1])
face3 = array(faces[2])

#get distances
momDistance = linalg.norm(face2-face3)
print ("Mom Distance:"+str(momDistance))

dadDistance = linalg.norm(face2-face1)
print ("Dad Distance:"+str(dadDistance))

#Print results
print()
print ("****************RESULTS****************")
if momDistance < dadDistance:
    print("You look more similar to your Dad")
elif momDistance > dadDistance:
    print ("You look more similar to your Mom")
else:
    print ("You look equally similar to both your mom and dad")
print()
