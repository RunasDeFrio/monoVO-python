import numpy as np 

class PointCloud:
    def __init__(self, X, Pt):
        self.X = X
        self.pt = Pt

class Frame:
    def __init__(self, frame, keypoints, descriptors):

        #image param
        self.frame = frame
        self.keypoints = keypoints
        self.descriptors = descriptors
        
        #the transformation matrix (global->local)
        self.R = np.eye( 3 ,  dtype= 'float64') 
        self.T = np.zeros(3)

        #containers
        self.key_map = {}
        self.track_key = set()
        self.space_point = {}

        self.triang_point = list()

    def GetP(self):
        self.T.reshape(3)
        return np.array(
            [[self.R[0][0],  self.R[0][1],    self.R[0][2],  self.T[0]],
            [self.R[1][0],  self.R[1][1],    self.R[1][2],  self.T[1]],
            [self.R[2][0],  self.R[2][1],    self.R[2][2],  self.T[2]],
            [      0, 	    0, 			0,     1]])

    #возвращает 3д координаты точкек и соотвествующие точки
    def GetVisibleSpacePoint(self):
        points = np.zeros((len(self.space_point), 2), dtype=np.float32)
        spaces = np.zeros((len(self.space_point), 3), dtype=np.float32)

        for i, j in enumerate(self.space_point.keys()):
            points[i, :] = self.keypoints[j].pt
            spaces[i, :] = self.space_point[j]

        return spaces, points

    #Записывает в словарь данные о меш-точках
    def mapingKeyPointsInFrame(self, matche, frame):
        for m in matche:
            self.key_map[m.trainIdx] = (m.queryIdx, frame)

            if(m.queryIdx in frame.track_key):
                self.track_key.add(m.trainIdx)
                self.space_point[m.trainIdx] = frame.space_point[m.queryIdx]
            else:
                self.triang_point.append(m.trainIdx)

    #Разрывает связи, если точка неправильная
    def SetSpacePoint(self, index, X, isTruePoint):
        if(isTruePoint):
            self.space_point[index] = X
            self.track_key.add(index)

            j, frame = self.key_map[index]

            frame.space_point[j] = X
            frame.track_key.add(j)
        else:
            self.key_map.pop(index)


    #!!!!НЕ ПРИМЕНЯТЬ ЕСЛИ БЫЛИ СВЯЗАННЫЕ ТОЧКИ С БОЛЕЕ ЧЕМ 1 ИЗОБРАЖЕНИЕМ!!!!
    def GetAllMatchKeyPoints(self):
		# Extract location of good self.matches

        points1 = np.zeros((len(self.key_map), 2), dtype=np.float32)
        points2 = np.zeros((len(self.key_map), 2), dtype=np.float32)

        for i, j in enumerate(self.key_map.keys()):
            k, frame = self.key_map[j]
            points1[i, :] = self.keypoints[j].pt
            points2[i, :] = frame.keypoints[k].pt

        return points2, points1
