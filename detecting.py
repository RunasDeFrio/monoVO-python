import numpy as np 
import cv2
import math

from point_cloud import Frame

kMinNumFeature = 1500
MAX_FEATURES = 2000
GOOD_MATCH_PERCENT = 0.15

detector = cv2.ORB_create(MAX_FEATURES, 1.2, 8, 31, 0, 2, 0, 31)
#detector = cv2.ORB_create(400, 5, 5)
#detector = cv2.xfeatures2d.SIFT_create()

class Detector:
	def __init__(self, frames, cam):
		self.detector = detector
		self.frames = frames
		self.matcher = cv2.DescriptorMatcher.create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT)
		self.cam = cam
	
	def Detect(self, frame):
		return self.detector.detectAndCompute(frame, None)
		
	def SortDistance(self, matches_all):
		min_d = 100
		for m in matches_all:
			if m.distance < min_d:
				min_d = m.distance

		matches = list()

		for m in matches_all:
			if(m.distance < 5*(min_d)):
				matches.append(m)
		return matches
		# Sort self.matches[-1] by score


	def DeleteEqual(self, matches_all):
		map_of_index = {}

		for m in matches_all:
			m2 = map_of_index.get(m.trainIdx)
			if(m2 != None):
				if(m.distance < m2.distance):
					map_of_index[m.trainIdx] = m
			else:
				map_of_index[m.trainIdx] = m
		matches = list(map_of_index.values()) 
		return matches
		# Sort self.matches[-1] by score



	def SortDelta(self, matches_all):
		matches = list()
		min_d = 100
		for m in matches_all:
			if(abs(self.frames[-2].keypoints[m.queryIdx].pt[0]-self.frames[-1].keypoints[m.trainIdx].pt[0])<(self.cam.width/3)):
				if(abs(self.frames[-2].keypoints[m.queryIdx].pt[1]-self.frames[-1].keypoints[m.trainIdx].pt[1])<(self.cam.height/5)):
					matches.append(m)
		return matches

	def SortLines(self, matches_all):
		def GetKB(x1, x2, y1, y2):
			dx = x1-x2
			dy = y1-y2
			

			if(dx == 0):
				k = 1000000
			else:
				k = dy/dx

			b = y1 - k*x1
			return (k, b)

		x0, y0 = None, None

		R = 100
		matches = list()
		Q = int(1000)
		traj = np.zeros((4*Q,4*Q,3), dtype=np.uint8)
		cv2.namedWindow('lines', cv2.WINDOW_NORMAL)

		sq = np.zeros((2,2), dtype = np.uint32)
		
		intersections = list()
		for i in range(len(matches_all)):
			m = matches_all[i]
			k0, b0 = GetKB(self.frames[-1].keypoints[m.trainIdx].pt[0], self.frames[-2].keypoints[m.queryIdx].pt[0], self.frames[-1].keypoints[m.trainIdx].pt[1], self.frames[-2].keypoints[m.queryIdx].pt[1])
			if(abs(b0) < 10*Q):
				cv2.line(traj, (int(Q-Q), int(-Q*k0+b0+Q)), (int(4*Q+Q), int(4*k0*Q+b0+Q)), (0,0,255), 1) #вывод пути камеры
			for j in range(i+1, len(matches_all)):
				k, b = GetKB(self.frames[-1].keypoints[matches_all[j].trainIdx].pt[0], self.frames[-2].keypoints[matches_all[j].queryIdx].pt[0], self.frames[-1].keypoints[matches_all[j].trainIdx].pt[1], self.frames[-2].keypoints[matches_all[j].queryIdx].pt[1])
				if(k0 == k):
					continue
				x = (b-b0)/(k0-k)
				y = k*x + b

				if(abs(x)> 6400 or abs(y)> 6400):
					continue
				cv2.circle(traj, (int(x+Q), int(y+Q)), 2, (0,255,0), 1)
				intersections.append((x, y))


		sq_size = 4*Q
		x0, y0 = -Q, -Q
		size_line = 0

		while (sq_size > 300 or (size_line/len(intersections)) > 0.6):
			for k in intersections:
				for i, row in enumerate(sq):
					for j, column in enumerate(row):
						xk = i*sq_size + sq_size/2 + x0
						yk = j*sq_size + sq_size/2 + y0
						if((abs(yk-k[1]) <= sq_size/2 )and (abs(xk-k[0]) <= sq_size/2)):
							sq[i, j] = sq[i, j] + 1
			i0, j0 = 0, 0
			for i, row in enumerate(sq):
				for j, column in enumerate(row):
					if(sq[i, j] > sq[i0, j0]):
						i0, j0 = i, j
						size_line = sq[i, j]
					text = str(sq[i, j])
					cv2.putText(traj, text, (int(i*sq_size + sq_size/2 + x0+Q),int(j*sq_size + sq_size/2 + y0+Q)), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
			for i, row in enumerate(sq):
				for j, column in enumerate(row):
					sq[i, j] = 0
			x0, y0 = i0*sq_size+x0, j0*sq_size+y0
			cv2.rectangle(traj,(int(x0+Q),int(y0+Q)),(int(x0+sq_size+Q),int(y0+sq_size+Q)),(128,128,0),2)
			sq_size = sq_size / 2
		
		sq_size = sq_size * 2
		cv2.rectangle(traj,(int(x0+Q),int(y0+Q)),(int(x0+sq_size+Q),int(y0+sq_size+Q)),(0,255,0),3)
		for m in matches_all:
			k, b = GetKB(self.frames[-1].keypoints[m.trainIdx].pt[0], self.frames[-2].keypoints[m.queryIdx].pt[0], self.frames[-1].keypoints[m.trainIdx].pt[1], self.frames[-2].keypoints[m.queryIdx].pt[1])
			xk = sq_size/2 + x0
			yk = sq_size/2 + y0
			y = k*xk + b
			if(abs(yk-y) <= sq_size/2):
				matches.append(m)
				#if(abs(b) < 10*Q):
					#cv2.line(traj, (int(0), int(b)), (int(Q), int(k*Q+b)), (0,128,255), 3) #вывод пути камеры

		print('count - ',size_line, '::', len(intersections))
		cv2.imshow('lines', traj)
		cv2.waitKey(1)
		return matches

	def SortAndDelete(self, matches_all):
		matches_all.sort(key=lambda x: x.distance, reverse=False)
		# Remove not so good self.matches[-1]
		numGoodMatches = int(len(smatches_all) * GOOD_MATCH_PERCENT)
		
		matches_all = matches_all[:numGoodMatches]
		
	def orbTracking(self, frame):

		key, descriptor = self.Detect(frame)

		self.frames.append(Frame(frame, key, descriptor))

		
		matches_all = self.matcher.match(self.frames[-2].descriptors, self.frames[-1].descriptors, None)
		
		matches_all = self.SortDelta(matches_all)
		matches_all = self.SortDistance(matches_all)
		matches_all = self.DeleteEqual(matches_all)

		#mat = self.SortLines(matches_all)

		#if(len(mat) > 0):
		#	matches_all = mat
		#self.matche = matches_all

		self.frames[-1].mapingKeyPointsInFrame(matches_all, self.frames[-2])
		return matches_all