import tensorflow as tf
import datetime
import requests
import subprocess
import os

# distance movement threshold
DISTANCE_THRESHOLD = 20.0
# time minimum threshold in seconds
# time max threshold in seconds
time_min, time_max = 1, 7

DOMAIN_AUTH_URL = "https://dashboard.back4app.com/login"
DOMAIN_AUTH_PAYLOAD = {
"user" : "mahendra.chhimwal@globalogic.com",
"password" : "Global@123"
}
validlist = [1,2,3,5,6,8]

CONFIDENCE_THRESHOLD = 0.1
dir = os.getcwd()

database = {
            "1": {"cart": {"body": '{"itemProduct": {"rating": "2.9", "title": "GL Grey Sports Tumbler Bottle", "imageUri": "https://image.ibb.co/gGfOgb/IMG_4928.jpg", "productObjectId": "QuEqpEmNas", "priceRegular": 1020, "priceSale": 1020, "productId": "561b83f5463cb072bea9dc22"}, "productObjectID": "QuEqpEmNas", "userObjectId": "a123XYZ", "image_url": "https://image.ibb.co/gGfOgb/IMG_4928.jpg", "title": "GL Grey Sports Tumbler Bottle"}', "url" : "https://parseapi.back4app.com/classes/Cart"},
                "history": {"body": '{"userObjectId": "a123XYZ", "title": "GL Grey Sports Tumbler Bottle", "imageUri": "https://image.ibb.co/gGfOgb/IMG_4928.jpg", "productObjectId": "QuEqpEmNas", "priceRegular": 1020, "priceSale": 1020, "productId": "561b83f5463cb072bea9dc22"}', "url": "https://parseapi.back4app.com/classes/UserBrowserHistory"},
                "video": os.path.join(dir,'productvideos','sipper.mov'),
                "DISTANCE_THRESHOLD":20,
                },
            "2":  {"cart": {"body": '{"itemProduct": {"rating": "4.8", "title": "GL Stainless Steel Travel Mug with Sipper Lid", "imageUri": "https://thumb.ibb.co/mSKS7G/IMG_4860.jpg", "productObjectId": "61FkQtu4pR", "priceRegular": 320, "priceSale": 320, "productId": "561b83f5463cb072bea9dc28"}, "productObjectID": "61FkQtu4pR", "userObjectId": "a123XYZ", "image_url": "https://thumb.ibb.co/mSKS7G/IMG_4860.jpg", "title": "GL Stainless Steel Travel Mug with Sipper Lid"}', "url" : "https://parseapi.back4app.com/classes/Cart"},
                "history": {"body": '{"userObjectId": "a123XYZ", "title": "GL Stainless Steel Travel Mug with Sipper Lid", "imageUri": "https://thumb.ibb.co/mSKS7G/IMG_4860.jpg", "productObjectId": "61FkQtu4pR", "priceRegular": 320, "priceSale": 320, "productId": "561b83f5463cb072bea9dc20"}', "url": "https://parseapi.back4app.com/classes/UserBrowserHistory"},
                "video": os.path.join(dir,'productvideos','pebblespeaker.mp4'),
                "DISTANCE_THRESHOLD":20,
                },
            "3": {"cart": {"body": '{"itemProduct": {"rating": "3.9", "title": "GL Classic Notebook", "imageUri": "https://image.ibb.co/mShQuw/IMG_4929.jpg", "productObjectId": "An2wVFGdTx", "priceRegular": 380, "priceSale": 380, "productId": "561b83f5463cb072bea9dc21"}, "productObjectID": "An2wVFGdTx", "userObjectId": "a123XYZ", "image_url": "https://image.ibb.co/mShQuw/IMG_4929.jpg", "title": "GL Classic Notebook"}', "url" : "https://parseapi.back4app.com/classes/Cart"},
                "history": {"body": '{"userObjectId": "a123XYZ", "title": "GL Classic Notebook", "imageUri": "https://image.ibb.co/mShQuw/IMG_4929.jpg", "productObjectId": "An2wVFGdTx", "priceRegular": 380, "priceSale": 380, "productId": "561b83f5463cb072bea9dc21"}', "url": "https://parseapi.back4app.com/classes/UserBrowserHistory"},
                "video": os.path.join(dir,'productvideos','diary.mov'),
                "DISTANCE_THRESHOLD":20,
                },
            "4": {"cart": {"body": '{"itemProduct": {"rating": "4.5", "title": "GL Hand Pressing Flash Light", "imageUri": "https://image.ibb.co/g6dWZw/IMG_4916.jpg", "productObjectId": "Mr3MgWhFMq", "priceRegular": 130, "priceSale": 130, "productId": "561b83f5463cb072bea9dc24"}, "productObjectID": "Mr3MgWhFMq", "userObjectId": "a123XYZ", "image_url": "https://image.ibb.co/g6dWZw/IMG_4916.jpg", "title": "GL Hand Pressing Flash Light"}', "url" : "https://parseapi.back4app.com/classes/Cart"},
			    "history": {"body": '{"userObjectId": "a123XYZ", "title": "GL Hand Pressing Flash Light", "imageUri": "https://image.ibb.co/g6dWZw/IMG_4916.jpg", "productObjectId": "Mr3MgWhFMq", "priceRegular": 130, "priceSale": 130, "productId": "561b83f5463cb072bea9dc22"}', "url": "https://parseapi.back4app.com/classes/UserBrowserHistory"},
                "video": os.path.join(dir,'productvideos','cap.mp4'),
                "DISTANCE_THRESHOLD":20,
                },
            "5": {"cart": {"body": '{"itemProduct": {"rating": "5", "title": "GL Penstand With Clock", "imageUri": "https://preview.ibb.co/cn5dEw/IMG_4855.jpg", "productObjectId": "JDaP5gNgl2", "priceRegular": 800, "priceSale": 800, "productId": "561b83f5463cb072bea9dc26"}, "productObjectID": "JDaP5gNgl2", "userObjectId": "a123XYZ", "image_url": "https://preview.ibb.co/cn5dEw/IMG_4855.jpg", "title": "GL Penstand With Clock"}', "url" : "https://parseapi.back4app.com/classes/Cart"},
                "history": {"body": '{"userObjectId": "a123XYZ", "title": "GL Penstand With Clock", "imageUri": "https://preview.ibb.co/cn5dEw/IMG_4855.jpg", "productObjectId": "JDaP5gNgl2", "priceRegular": 800, "priceSale": 800, "productId": "561b83f5463cb072bea9dc20"}', "url": "https://parseapi.back4app.com/classes/UserBrowserHistory"},
                "video": os.path.join(dir,'productvideos','greenbottle.mp4'),
                "DISTANCE_THRESHOLD":20,
                },
            "6": {"cart": {"body": '{"itemProduct": {"rating": "2.9", "title": "Shantnu & Nikhil Key Chain", "imageUri": "https://image.ibb.co/c0CfSG/IMG_4888.jpg", "productObjectId": "bFay3Sz6yD", "priceRegular": 400, "priceSale": 400, "productId": "561b83f5463cb072bea9dc23"}, "productObjectID": "bFay3Sz6yD", "userObjectId": "a123XYZ", "image_url": "https://image.ibb.co/c0CfSG/IMG_4888.jpg", "title": "Shantnu & Nikhil Key Chain"}', "url" : "https://parseapi.back4app.com/classes/Cart"},
			    "history": {"body": '{"userObjectId": "a123XYZ", "title": "Shantnu & Nikhil Key Chain", "imageUri": "https://image.ibb.co/c0CfSG/IMG_4888.jpg", "productObjectId": "bFay3Sz6yD", "priceRegular": 400, "priceSale": 400, "productId": "561b83f5463cb072bea9dc23"}', "url": "https://parseapi.back4app.com/classes/UserBrowserHistory"},
                "video": os.path.join(dir,'productvideos','shantanukeychain.mov'),
                "DISTANCE_THRESHOLD":7,
                },
            "7": {"cart": {"body": '{"itemProduct": {"rating": "4.2", "title": "Matte Black Ballpoint Pen", "imageUri": "https://image.ibb.co/ddvZnG/IMG_4842.jpg", "productObjectId": "kpCgBGbw13", "priceRegular": 3180, "priceSale": 3180, "productId": "561b83f5463cb072bea9dc20"}, "productObjectID": "kpCgBGbw13", "userObjectId": "a123XYZ", "image_url": "https://image.ibb.co/ddvZnG/IMG_4842.jpg", "title": "Matte Black Ballpoint Pen"}', "url" : "https://parseapi.back4app.com/classes/Cart"},
			    "history": {"body": '{"userObjectId": "a123XYZ", "title": "Matte Black Ballpoint Pen", "imageUri": "https://image.ibb.co/ddvZnG/IMG_4842.jpg", "productObjectId": "kpCgBGbw13", "priceRegular": 3180, "priceSale": 3180, "productId": "561b83f5463cb072bea9dc20"}', "url": "https://parseapi.back4app.com/classes/UserBrowserHistory"},
                "video": os.path.join(dir,'productvideos','redsipper.mp4'),
                "DISTANCE_THRESHOLD":20,
                },
            "8": {"cart": {"body": '{"itemProduct": {"rating": "2.9", "title": "GL Bluetooth Headphones", "imageUri": "https://image.ibb.co/gGfOgb/IMG_4928.jpg", "productObjectId": "GR8FeLeV5M", "priceRegular": 2425, "priceSale": 2425, "productId": "561b83f5463cb072bea9dc25"}, "productObjectID": "GR8FeLeV5M", "userObjectId": "a123XYZ", "image_url": "https://image.ibb.co/gGfOgb/IMG_4928.jpg", "title": "GL Bluetooth Headphones"}', "url" : "https://parseapi.back4app.com/classes/Cart"},
			    "history": {"body": '{"userObjectId": "a123XYZ", "title": "GL Bluetooth Headphones", "imageUri": "https://image.ibb.co/gGfOgb/IMG_4928.jpg", "productObjectId": "GR8FeLeV5M", "priceRegular": 2425, "priceSale": 2425, "productId": "561b83f5463cb072bea9dc25"}', "url": "https://parseapi.back4app.com/classes/UserBrowserHistory"},
                "video": os.path.join(dir,'productvideos','headphone.mov'),
                "DISTANCE_THRESHOLD":20,
                },
            }

headers = {
"X-Parse-Application-Id": "APUBPWnDEekm3ar8ZqwAVAWgqrG8ooe2pXyMlt9S",
"X-Parse-REST-API-Key": "xUAT1d8L5rP8nEjLrDBdH9KxIxAuoJGQNz3QxYQV",
"Content-Type" : "application/json"
}



class DataCast:
    class __DataCast:
        """privately maintained singletone class"""
        def __init__(self, arg):
            self.name = arg[6]
            self.val = arg[4]
            self.primary = arg
            self.initTime = arg[5]
            self.curr_arg= None
            self.played = False
            self.picked = False
        def __str__(self):
            return repr(self) + self.val

        def center(self,arg):
            # calculate the center of the coordinates
            ords1 = self.arg[0:4]
            ycenter = (ords[0]+ords[2])/2
            xcenter = (ords[1]+ ords[3])/2
            return(ycenter,xcenter)

        def get_distance(self, arg):
            # calculate distance travelled now
            ords = self.primary[0:4]
            ycenter = (ords[0]+ords[2])/2
            xcenter = (ords[1]+ ords[3])/2
            ords1 = (ycenter, xcenter)
            ords = arg[0:4]
            ycenter = (ords[0]+ords[2])/2
            xcenter = (ords[1]+ ords[3])/2
            ords2 = (ycenter, xcenter)
            distance = ((ords2[0]-ords1[0])**2 +(ords2[1]-ords1[1])**2)**0.5
            return distance

        def get_timediff(self, arg):
            time1 = self.initTime
            time2 = arg[5]
            timed = time2 - time1
            total = 0
            total+= timed.days * (60*60*60)
            total+= timed.seconds
            return total

        def evaluate(self, dist, times, arg):
            self.curr_arg = arg             
            dist = int(dist)
            if arg[4] in validlist:
                DISTANCE_THRESHOLD = int(database[str(self.val)]["DISTANCE_THRESHOLD"])                
                print("%s => %s / %s" %(str(self.primary[6]),dist,DISTANCE_THRESHOLD))
                print("Elapsed=> %s , played => %s , picked=> %s" %(times, self.played,self.picked))
                if dist >= DISTANCE_THRESHOLD and self.played == False:  
                    print("running video for %s =>" %self.primary[6])                  
                    self.run_video(self.primary[6],database[str(self.val)]["video"])
                    self.initTime = datetime.datetime.now()
                    if self.picked== False:
                        print("calling history post for %s =>" %self.primary[6])
                        # self.call_url(str(self.val), "history")
                        self.picked = True
                if times > time_max and self.played == True and dist <= DISTANCE_THRESHOLD:
                    print("set play to false for %s =>" %self.primary[6])
                    self.played = False        
                if times > time_max and self.picked == True:
                    self.initTime = datetime.datetime.now()                
                    print("Sending to cart for %s =>" %self.primary[6])
                    # DataCast.instance[ct].call_url(str(ct), "cart")
                    self.picked = False                                           

            def call_url(self, classno, action):
                with requests.session() as s:
                    l_res = s.post(DOMAIN_AUTH_URL, data=DOMAIN_AUTH_PAYLOAD)
                    if not l_res.ok:
                        print('resource not connected ...')
                    # now post the body
                    url = database[classno][action]["url"]
                    body= database[classno][action]["body"]
                    s.post(url, headers=headers, data=body)
                    print ("post for image_class %s, action %s" %(classno, action))
                    print(res2)                    

        def run_video(self, classno, video):
            print("running video for %s ..." %classno)
            if self.played is False:
                subprocess.Popen('"C:\\Program Files (x86)\\VideoLAN\\VLC\\vlc.exe" --no-qt-name-in-title -f "%s"'%video,
                stderr=subprocess.STDOUT,
                shell=True)
                self.played = True
            # now add the class to browsing history

    instance = {}

    def __init__(self, arg):
        # populate the instance dictionary with key
        self.class_type = arg[4]
        if self.class_type not in DataCast.instance.keys():
            DataCast.instance[self.class_type] = DataCast.__DataCast(arg)            
        else:
            distance_travelled = DataCast.instance[self.class_type].get_distance(arg)            
            timelapsed = DataCast.instance[self.class_type].get_timediff(arg)            
            DataCast.instance[self.class_type].evaluate(distance_travelled, timelapsed, arg)

    def __getattr__(self, name):
        # print("class_type=> %s" %self.class_type)
        return getattr(self.instance[self.class_type], name)


def pack_objs(image,classnames,boxes, classes, confidence,maxresults):
    im_height, im_width = image.shape[:2]
    current_time = datetime.datetime.now()
    packed_dims = []    
    for i in range(0,maxresults):
        if confidence[i] > CONFIDENCE_THRESHOLD:
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            (left, right, top, bottom) = (round(xmin * im_width), round(xmax * im_width),
                                  round(ymin * im_height), round(ymax * im_height))           
            image_class = classes[i]
            classname =classnames[classes[i]]["name"]
            data =(left,right,top,bottom, image_class, current_time,classname)
            # print('%s=> left:%d right:%d top:%d bottom:%d'%(classname,left,right,top,bottom))
            packed_dims.append(data)
    return packed_dims    

def map_objs(packed_dims):    
    # class shall manage the object consumption
    for items in packed_dims:
        DataCast(items)