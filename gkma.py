import sys
import random
import copy
from functools import cmp_to_key

class VertexInfo:
	def __init__(self, clusterIndex, vertexIndex):
		self.clusterIndex = clusterIndex
		self.vertexIndex = vertexIndex

class Task:
	def __init__(self):
		self.m = None
		self.clusters = None
		self.clusterCount = None
		self.clusterByVertex = None
		self.verticesInfo = None
		self.verticeCount = None
		self.minClusterDistances = None
	def GetCluster2(self, vertex):
		return self.verticesInfo[vertex].clusterIndex
	def GetCluster(self, vertex):
		return self.verticesInfo[vertex].clusterIndex, self.verticesInfo[vertex].vertexIndex

task = Task()

def Weight(v):
	value = 0
	for i in range(len(v) - 1):
		value += task.m[v[i]][v[i + 1]]
	return value

def MinClusterDistance(v):
	value = 0
	for i in range(len(v) - 1):
		value += task.minClusterDistances[v[i]][v[i + 1]]
	return value

def MinClusterDistanceForVertices(v):
	return MinClusterDistance([task.clusterByVertex[i] for i in v])
	
def PrevPos(pos):
	if pos == 0:
		return task.clusterCount - 1
	return pos - 1

def NextPos(pos):
	if pos == task.clusterCount - 1:
		return 0
	return pos + 1

def Rotate(tour, firstClusterPosition):
	newTour = [0 for i in range(len(tour))]
	for i in range(len(tour) - firstClusterPosition):
		newTour[i] = tour[firstClusterPosition + i]
	for i in range(firstClusterPosition):
		newTour[len(tour) - firstClusterPosition + i] = tour[i]
	return newTour

def FullSwap(solution):
	totalDelta = 0
	next2Pos = 0
	for pos2 in range(task.clusterCount - 1, 1, -1):
		prev2Pos = pos2 - 1
		prev2 = solution[prev2Pos]
		cur2 = solution[pos2]
		next2 = solution[next2Pos]
		for pos1 in range(pos2 - 2, 0, -1):
			prev1Pos = pos1 - 1
			prev1 = solution[prev1Pos]
			cur1 = solution[pos1]
			next1Pos = pos1 + 1
			next1 = solution[next1Pos]
			delta = Weight([prev1, cur2, next1]) + Weight([prev2, cur1, next2]) - Weight([prev1, cur1, next1]) - Weight([prev2, cur2, next2])
			if delta < 0:
				solution[pos1] = cur2
				solution[pos2] = cur1
				cur2 = solution[pos2]
				totalDelta += delta
		next2Pos = pos2
	return totalDelta

def InsertsWithCO(solution):
	def Insert(solution, oldPosition, newPosition):
		oldV = solution[oldPosition]
		if oldPosition < newPosition:
			for i in range(oldPosition, newPosition):
				solution[i] = solution[i + 1]
		else:
			for i in range(oldPosition, newPosition, -1):
				solution[i] = solution[i - 1]
		solution[newPosition] = oldV
	totalDelta = 0
	for oldPos in range(task.clusterCount - 1):
		oldV = solution[oldPos]
		oldPrevPos = PrevPos(oldPos)
		oldPrev = solution[oldPrevPos]
		oldNextPos = NextPos(oldPos)
		oldNext = solution[oldNextPos]

		cluster = task.clusterByVertex[oldV]

		oldWeightFirstPartBefore = Weight([oldPrev, oldV, oldNext])
		oldWeightFirstPartAfter = Weight([oldPrev, oldNext])

		for newPos in range(oldPos - 1):
			newPrevPos = PrevPos(newPos)
			newNextPos = newPos

			newPrev = solution[newPrevPos]
			newNext = solution[newNextPos]

			oldWeight = oldWeightFirstPartBefore + Weight([newPrev, newNext])
			if oldWeightFirstPartAfter + MinClusterDistanceForVertices([newPrev, oldV, newNext]) >= oldWeight:
				continue

			minL = sys.maxsize
			bestNewV = 0
			for i in range(len(task.clusters[cluster]) - 1, -1, -1):
				newV = task.clusters[cluster][i]
				l = Weight([newPrev, newV, newNext])
				if l < minL:
					minL = l
					bestNewV = newV

			delta = oldWeightFirstPartAfter + minL - oldWeight
			if delta >= 0:
				continue

			Insert(solution, oldPos, newPos)
			solution[newPos] = bestNewV

			oldV = solution[oldPos]
			oldPrev = solution[oldPrevPos]
			oldNext = solution[oldNextPos]
			cluster = task.clusterByVertex[oldV]
			oldWeightFirstPartBefore = Weight([oldPrev, oldV, oldNext])
			oldWeightFirstPartAfter = Weight([oldPrev, oldNext])

			totalDelta += delta

		for newPos in range(oldPos + 1, task.clusterCount - 1):
			newPrevPos = newPos
			newNextPos = newPos + 1

			newPrev = solution[newPrevPos]
			newNext = solution[newNextPos]

			oldWeight = oldWeightFirstPartBefore + Weight([newPrev, newNext])
			if oldWeightFirstPartAfter + MinClusterDistanceForVertices([newPrev, oldV, newNext]) >= oldWeight:
				continue

			minL = sys.maxsize
			bestNewV = 0
			for i in range(len(task.clusters[cluster]) - 1, -1, -1):
				newV = task.clusters[cluster][i]
				l = Weight([newPrev, newV, newNext])
				if l < minL:
					minL = l
					bestNewV = newV

			delta = oldWeightFirstPartAfter + minL - oldWeight
			if delta >= 0:
				continue

			Insert(solution, oldPos, newPos)
			solution[newPos] = bestNewV

			oldV = solution[oldPos]
			oldPrev = solution[oldPrevPos]
			oldNext = solution[oldNextPos]
			cluster = task.clusterByVertex[oldV]
			oldWeightFirstPartBefore = Weight([oldPrev, oldV, oldNext])
			oldWeightFirstPartAfter = Weight([oldPrev, oldNext])

			totalDelta += delta
	return totalDelta

def TwoOptFullAsym(solution):
	def RestoreLargePos(largePos):
		if largePos >= task.clusterCount:
			return largePos - task.clusterCount
		return largePos

	totalDelta = 0
	for begin2pos in range(task.clusterCount):
		reverseDelta = 0
		end1pos = PrevPos(begin2pos)
		for len in range(2, task.clusterCount - 1):
			end2pos = RestoreLargePos(begin2pos + len - 1)
			begin1pos = NextPos(end2pos)
			prevEnd2pos = PrevPos(end2pos)
			reverseDelta += Weight([solution[end2pos], solution[prevEnd2pos]]) - Weight([solution[prevEnd2pos], solution[end2pos]])
			delta = Weight([solution[end1pos], solution[end2pos]]) + Weight([solution[begin2pos], solution[begin1pos]]) - Weight([solution[end1pos], solution[begin2pos]]) - Weight([solution[end2pos], solution[begin1pos]]) + reverseDelta

			if delta < 0:
				totalDelta += delta
				reverseDelta = -reverseDelta
				p1 = begin2pos
				p2 = end2pos
				for i in range(len // 2):
					temp = solution[p1]
					solution[p1] = solution[p2]
					solution[p2] = temp

					p1 = NextPos(p1)
					p2 = PrevPos(p2)
	return totalDelta

def NeighbourSwapWithCO(solution):
	def NeighbourSwapWithClusterOptimisation(prevVertex, v1, v2, nextVertex):
		oldLen = Weight([prevVertex, v1, v2, nextVertex])
		cluster1 = task.clusterByVertex[v1]
		cluster2 = task.clusterByVertex[v2]
		bestV1 = 0
		bestV2 = 0

		if MinClusterDistance([task.clusterByVertex[prevVertex], cluster2, cluster1, task.clusterByVertex[nextVertex]]) >= oldLen:
			return 0, v1, v2

		minDelta = 0
		for v1index in range(len(task.clusters[cluster1]) - 1, -1, -1):
			curV2 = task.clusters[cluster1][v1index]
			for v2index in range(len(task.clusters[cluster2]) - 1, -1, -1):
				curV1 = task.clusters[cluster2][v2index]
				lens = Weight([prevVertex, curV1, curV2, nextVertex])
				delta = lens - oldLen
				if delta < minDelta:
					minDelta = delta
					bestV1 = curV1
					bestV2 = curV2
		if minDelta < 0:
			v1 = bestV1
			v2 = bestV2
		return minDelta, v1, v2
	totalDelta = 0
	prevPos = task.clusterCount - 3
	v1Pos = task.clusterCount - 2
	v2Pos = task.clusterCount - 1

	for nextPos in range(task.clusterCount):
		delta, v1, v2 = NeighbourSwapWithClusterOptimisation(solution[prevPos], solution[v1Pos], solution[v2Pos], solution[nextPos])
		totalDelta += delta
		solution[v1Pos] = v1
		solution[v2Pos] = v2

		prevPos = v1Pos
		v1Pos = v2Pos
		v2Pos = nextPos
	return totalDelta

def DirectTwoOptAsym(solution):
	def InsertEdge(edges, edgeArraySize, pos2, w):
		edges[0]['pos2'] = pos2
		edges[0]['length'] = w
		for i in range(1, edgeArraySize):
			if edges[i - 1]['length'] > edges[i]['length']:
				temp = edges[i]
				edges[i] = edges[i - 1]
				edges[i - 1] = temp
	def CompareEdges(edge1, edge2):
		if edge1['length'] < edge2['length']:
			return -1
		if edge1['length'] > edge2['length']:
			return 1
		return 0
	def RestoreSmallPos(smallPos):
		if smallPos < 0:
			return smallPos + task.clusterCount
		return smallPos
	n = task.clusterCount // 4
	edges = [{'pos2':0, 'length':0} for i in range(n)]

	prevV = solution[task.clusterCount - 1]
	for i in range(n):
		curV = solution[i]
		edges[i]['pos2'] = i
		edges[i]['length'] = task.m[prevV][curV]
		prevV = curV

	edges.sort(key=cmp_to_key(CompareEdges))

	for i in range(n, task.clusterCount):
		curV = solution[i]
		w = task.m[prevV][curV]
		if w > edges[0]['length']:
			InsertEdge(edges, n, i, w)

	totalDelta = 0
	for begin2Index in range(n - 1, 0, -1):
		begin2pos = edges[begin2Index]['pos2']
		end1pos = PrevPos(begin2pos)
		end1 = solution[end1pos]
		begin2 = solution[begin2pos]

		for begin1Index in range(begin2Index - 1, -1, -1):
			begin1pos = edges[begin1Index]['pos2']
			end2pos = PrevPos(begin1pos)
			if begin1pos == end1pos or begin2pos == end2pos:
				continue

			begin1 = solution[begin1pos]
			end2 = solution[end2pos]

			delta = task.m[end1][end2] + task.m[begin2][begin1] - task.m[end1][begin2] - task.m[end2][begin1]

			p = begin2pos
			while p != end2pos:
				delta -= Weight([solution[p], solution[NextPos(p)]]) - Weight([solution[NextPos(p)], solution[p]])
				p = NextPos(p)

			if delta < 0:
				p1 = begin2pos
				p2 = end2pos
				lens = RestoreSmallPos(end2pos - begin2pos) + 1
				
				for i in range(lens // 2, 0, -1):
					temp = solution[p1]
					solution[p1] = solution[p2]
					solution[p2] = temp
					p1 = NextPos(p1)
					p2 = PrevPos(p2)

				totalDelta += delta

				end1 = solution[end1pos]
				begin2 = solution[begin2pos]
	return totalDelta

def ThreeNeighbourFullSwap(solution):
	def TrySwap(prev, c1, c2, c3, next, vertices, oldDist):
		prevC = task.clusterByVertex[prev]
		nextC = task.clusterByVertex[next]

		if MinClusterDistance([prevC, c1, c2, c3, nextC]) >= oldDist:
			return 0
		c1Size = len(task.clusters[c1])
		c2Size = len(task.clusters[c2])
		c3Size = len(task.clusters[c3])

		minTotal = oldDist
		bestV1 = 0
		bestV2 = 0
		bestV3 = 0
		for v2Index in range(c2Size - 1, -1, -1):
			curV2 = task.clusters[c2][v2Index]
			leftMin = sys.maxsize
			bestV1forCurV2 = 0
			for v1Index in range(c1Size - 1, -1, -1):
				curV1 = task.clusters[c1][v1Index]
				left = Weight([prev, curV1, curV2])
				if left < leftMin:
					leftMin = left
					bestV1forCurV2 = curV1

			rightMin = sys.maxsize
			bestV3forCurV2 = 0
			for v3Index in range(c3Size - 1, -1, -1):
				curV3 = task.clusters[c3][v3Index]
				right = Weight([curV2, curV3, next])
				if right < rightMin:
					rightMin = right
					bestV3forCurV2 = curV3

			if leftMin + rightMin < minTotal:
				bestV1 = bestV1forCurV2
				bestV2 = curV2
				bestV3 = bestV3forCurV2
				minTotal = leftMin + rightMin

		delta = minTotal - oldDist
		if delta < 0:
			vertices[0] = bestV1
			vertices[1] = bestV2
			vertices[2] = bestV3
		return delta
	totalDelta = 0

	prevPos = task.clusterCount - 4
	pos1 = task.clusterCount - 3
	pos2 = task.clusterCount - 2
	pos3 = task.clusterCount - 1
	for nextPos in range(task.clusterCount - 1):
		prev = solution[prevPos]
		v1 = solution[pos1]
		v2 = solution[pos2]
		v3 = solution[pos3]
		next = solution[nextPos]

		c1 = task.clusterByVertex[v1]
		c2 = task.clusterByVertex[v2]
		c3 = task.clusterByVertex[v3]

		oldDist = Weight([prev, v1, v2, v3, next])

		vertices1 = [0 for i in range(3)]
		vertices2 = [0 for i in range(3)]
		vertices3 = [0 for i in range(3)]
		delta1 = TrySwap(prev, c2, c3, c1, next, vertices1, oldDist)
		delta2 = TrySwap(prev, c3, c1, c2, next, vertices2, oldDist)
		delta3 = TrySwap(prev, c3, c2, c1, next, vertices3, oldDist)
		if delta1 < 0 or delta2 < 0 or delta3 < 0:
			if delta1 <= delta2 and delta1 <= delta3:
				vertices = vertices1
				totalDelta += delta1
			elif delta2 <= delta1 and delta2 <= delta3:
				vertices = vertices2
				totalDelta += delta2
			elif delta3 <= delta1 and delta3 <= delta2:
				vertices = vertices3
				totalDelta += delta3
			solution[pos1] = vertices[0]
			solution[pos2] = vertices[1]
			solution[pos3] = vertices[2]
		prevPos = pos1
		pos1 = pos2
		pos2 = pos3
		pos3 = nextPos
	return totalDelta

def ClusterOptimisation(result):
	def FindBestPath(firstClusterVertex):
		bestCurLen = [0 for i in range(task.verticeCount)]
		bestPrevLen = [0 for i in range(task.verticeCount)]

		toCluster = clusterSequence[1]
		for i in range(len(task.clusters[toCluster])):
			toVertex = task.clusters[toCluster][i]
			bestPrevLen[i] = task.m[firstClusterVertex][toVertex]
			bestFrom[toVertex] = firstClusterVertex

		for pos in range(2, task.clusterCount):
			toCluster = clusterSequence[pos]
			fromCluster = clusterSequence[pos - 1]
			for toVertexIndex in range(len(task.clusters[toCluster])):
				toVertex = task.clusters[toCluster][toVertexIndex]
				bestCurLen[toVertexIndex] = sys.maxsize
				for fromVertexIndex in range(len(task.clusters[fromCluster])):
					fromVertex = task.clusters[fromCluster][fromVertexIndex]
					lens = bestPrevLen[fromVertexIndex] + task.m[fromVertex][toVertex]
					if lens < bestCurLen[toVertexIndex]:
						bestCurLen[toVertexIndex] = lens
						bestFrom[toVertex] = fromVertex
			temp = bestCurLen
			bestCurLen = bestPrevLen
			bestPrevLen = temp

		fromCluster = clusterSequence[task.clusterCount - 1]
		minLen = sys.maxsize
		for i in range(len(task.clusters[fromCluster]) - 1, -1, -1):
			fromVertex = task.clusters[fromCluster][i]
			lens = bestPrevLen[i] + task.m[fromVertex][firstClusterVertex]
			if lens < minLen:
				minLen = lens
				bestFrom[firstClusterVertex] = fromVertex
		return minLen
	def RestoreSolution(vertexIndices, firstVertex):
		prev = firstVertex
		for i in range(task.clusterCount - 1, -1, -1):
			prev = bestFrom[prev]
			vertexIndices[i] = prev
	def FindSmallestClusterPosition(solution):
		minSize = sys.maxsize
		bestPos = 0
		for i in range(task.clusterCount):
			vertex = solution[i]
			cluster = task.clusterByVertex[vertex]
			clusterSize = len(task.clusters[cluster])
			if clusterSize < minSize:
				if clusterSize == 1:
					return i
				minSize = clusterSize
				bestPos = i
		return bestPos
	clusterSequence = [0 for i in range(task.clusterCount)]
	bestFrom = [0 for i in range(task.verticeCount)]
	
	firstPosition = FindSmallestClusterPosition(result)
	index = 0
	for i in range(firstPosition, task.clusterCount):
		clusterSequence[index] = task.clusterByVertex[result[i]]
		index += 1
	for i in range(firstPosition):
		clusterSequence[index] = task.clusterByVertex[result[i]]
		index += 1

	minLen = sys.maxsize
	firstCluster = clusterSequence[0]
	bestSolution = [0 for i in range(task.clusterCount)]
	for i in range(len(task.clusters[firstCluster]) - 1, -1, -1):
		firstVertex = task.clusters[firstCluster][i]
		lens = FindBestPath(firstVertex)
		if lens < minLen:
			if i == 0:
				RestoreSolution(result, firstVertex)
				return lens
			else:
				minLen = lens
				RestoreSolution(bestSolution, firstVertex)
	for i in range(task.clusterCount):
		result[i] = bestSolution[i]
	return minLen

class Generation:
	def __init__(self, generationIndex):
		self.generationIndex = generationIndex
		self.tours = []
	def Sort(self):
		def CompareTo(g1, g2):
			if g1.length < g2.length:
				return -1
			if g1.length > g2.length:
				return 1
			return 0
		self.tours.sort(key=cmp_to_key(CompareTo))
	def Contains(self, tour):
		return tour in self.tours
	def Add(self, tour):
		if tour not in self.tours:
			self.tours.append(tour)
	def Size(self):
		return len(self.tours)

class Permutation:
	def __init__(self, n):
		self.values = [0 for i in range(n)]
		for i in range(n):
			self.values[i] = i
	def Swap(self, index1, index2):
		t = self.values[index1]
		self.values[index1] = self.values[index2]
		self.values[index2] = t

class RandomGenerate:
	def GenerateKey(self):
		def GenerateRandom(n):
			result = Permutation(n)
			for i in range(n - 1):
				result.Swap(i, random.randint(i, n - 1))
			return result
				
		permutation = GenerateRandom(task.clusterCount)
		result = [0 for i in range(task.clusterCount)]
		for i in range(task.clusterCount):
			result[i] = task.clusters[permutation.values[i]][random.randint(0, len(task.clusters[permutation.values[i]]) - 1)]
		ClusterOptimisation(result)
		return result

class RandomTourItem:
	def __init__(self, clusterIndex, value):
		self.clusterIndex = clusterIndex
		self.vertexInCluster = int(value)
		self.valueFrac = value - self.vertexInCluster

class TourElement:
	def __init__(self, clusterIndex, vertexInCluster):
		self.clusterIndex = clusterIndex
		self.vertexInCluster = vertexInCluster
	def GetVertexIndex(self):
		return task[self.clusterIndex][self.vertexInCluster]

FullSwapId = 1
InsertsWithCoId = 2
DirectTwoOptId = 4
TwoOptId = 8
NeighbourSwapWithCoId = 16
ThreeNeighbourFullSwapId = 32
H = [
	[ FullSwap,					FullSwapId,					FullSwapId ],
	[ InsertsWithCO,			InsertsWithCoId,			InsertsWithCoId ],
	[ DirectTwoOptAsym,			DirectTwoOptId,				DirectTwoOptId | TwoOptId ],
	[ TwoOptFullAsym,			TwoOptId,					TwoOptId ],
	[ NeighbourSwapWithCO,		NeighbourSwapWithCoId,		NeighbourSwapWithCoId ],
	[ ThreeNeighbourFullSwap,	ThreeNeighbourFullSwapId,	ThreeNeighbourFullSwapId ],
]

class Tour:
	def __init__(self, generation, strategy = None, vertices = None):
		def crossoverCross(parent1, parent2):
			result = [0 for i in range(task.clusterCount)]

			point1 = random.randint(0, task.clusterCount - 1)
			while True:
				point2 = random.randint(0, task.clusterCount - 1)
				if point1 != point2:
					break

			sourcePosition1 = point1
			sourcePosition2 = point1
			exists = [False for i in range(task.clusterCount)]
			for i in range(task.clusterCount):
				while exists[parent1.GetCluster(sourcePosition1)]:
					sourcePosition1 = NextPos(sourcePosition1)

				result[i] = parent1.vertices[sourcePosition1]
				exists[parent1.GetCluster(sourcePosition1)] = True

				if i == parent1.RestoreIndex(point2 - point1):
					tempParent = parent1
					parent1 = parent2
					parent2 = tempParent

					tempSourcePosition = sourcePosition1
					sourcePosition1 = sourcePosition2
					sourcePosition2 = tempSourcePosition
			return result

		if strategy:
			self.firstClusterPosition = -1
			index1, index2 = strategy.Run(generation)
			parent1 = generation.tours[index1]
			parent2 = generation.tours[index2]

			parent1.CorrectRotation()
			parent2.CorrectRotation()

			self.vertices = crossoverCross(parent1, parent2)
			self.length = -1
			self.UpdateLength()
		elif vertices:
			self.firstClusterPosition = -1
			self.vertices = vertices
			self.length = -1
			self.UpdateLength()
		else:
			self.firstClusterPosition = -1
			self.vertices = generation.GenerateKey()
			self.length = -1
			self.UpdateLength()
	def UpdateLength(self):
		self.length = task.m[self.vertices[len(self.vertices) - 1]][self.vertices[0]]
		for i in range(1, len(self.vertices)):
			self.length += task.m[self.vertices[i - 1]][self.vertices[i]]
	def RestoreIndex(self, index):
		if index < 0:
			return index + task.clusterCount
		if index >= task.clusterCount:
			return index - task.clusterCount
		return index
	def GetCluster(self, position):
		return task.GetCluster2(self.vertices[position])
	def GetElement(self, position):
		cluster, vertexIndex = task.GetCluster(self.vertices[position])
		item = RandomTourItem(cluster, vertexIndex)
		return TourElement(item.clusterIndex, item.vertexInCluster)
	def GetFirstClusterPosition(self):
		if self.firstClusterPosition >= 0:
			return self.firstClusterPosition

		for i in range(task.clusterCount):
			if self.GetElement(i).clusterIndex == 0:
				self.firstClusterPosition = i
				return self.firstClusterPosition
		return self.firstClusterPosition
	def CorrectRotation(self):
		if self.GetFirstClusterPosition() == 0:
			return
		self.vertices = Rotate(self.vertices, self.firstClusterPosition)
		self.firstClusterPosition = 0
	def Improve(self):
		onceFailedHeuristics = 0
		totalDelta = 0
		while True:
			idleCycle = True
			for i in range(len(H)):
				if onceFailedHeuristics & H[i][2]:
					continue
				delta = H[i][0](self.vertices)
				if delta < 0:
					idleCycle = False
					totalDelta += delta
				else:
					onceFailedHeuristics |= H[i][1]
			if idleCycle:
				break
		self.length = ClusterOptimisation(self.vertices)

class ElitistTwoStrategy:
	def __init__(self, part):
		self.part = part
	def Run(self, generation):
		n = int(generation.Size() * self.part)
		index1 = random.randint(0, n - 1)
		while True:
			index2 = random.randint(0, generation.Size() - 1)
			if index1 != index2:
				break
		return index1, index2
crossoverStrategy = ElitistTwoStrategy(0.33)

class ElitistOneStrategy:
	def __init__(self, max):
		self.max = max
	def Run(self, generation):
		maxIndex = int(generation.Size() * self.max)
		return int((random.uniform(0, 1) ** 2) * maxIndex)
mutationStrategy = ElitistOneStrategy(0.75)
	
class Algorithm:
	def __init__(self, m, clusters):
		def CalculateMinClusterDistance(fromCluster, toCluster):
			min = sys.maxsize
			for fromVertexIndex in range(len(task.clusters[fromCluster])):
				fromVertex = task.clusters[fromCluster][fromVertexIndex]
				for toVertexIndex in range(len(task.clusters[toCluster])):
					toVertex = task.clusters[toCluster][toVertexIndex]
					w = task.m[fromVertex][toVertex]
					if w < min:
						min = w
			return min
		self.prev = None
		self.generationCount = 0
		self.consecutiveIdleGenerationCount = 0
		self.maxIdleGenerations = 0
		
		infinite = int(sys.maxsize / len(clusters))
		task.verticeCount = len(m)
		for i in range(task.verticeCount):
			for j in range(task.verticeCount):
				if m[i][j] == -1:
					m[i][j] = infinite
				else:
					m[i][j] = int(m[i][j] * 1000)
		task.m = m
		
		task.clusters = clusters
		task.clusterCount = len(clusters)
		task.clusterByVertex = [0 for i in range(task.verticeCount)]
		task.verticesInfo = [0 for i in range(task.verticeCount)]
		for i in range(task.clusterCount):
			for j in range(len(clusters[i])):
				task.clusterByVertex[clusters[i][j]] = i
				task.verticesInfo[clusters[i][j]] = VertexInfo(i, j)
				
		task.minClusterDistances = [[0 for i in range(task.clusterCount)] for j in range(task.clusterCount)]
		for fromCluster in range(task.clusterCount):
			for toCluster in range(task.clusterCount):
				if toCluster != fromCluster:
					task.minClusterDistances[fromCluster][toCluster] = CalculateMinClusterDistance(fromCluster, toCluster)
	def MaxTriesForFirstGeneration(self):
		return task.clusterCount * 4
	def FirstGeneration(self, cur):
		generatingAlgorithm = RandomGenerate()
		for tries in range(self.MaxTriesForFirstGeneration()):
			tour = Tour(generatingAlgorithm)
			tour.Improve()
			if not cur.Contains(tour):
				cur.Add(tour)
		cur.Sort()
	def AddGeneration(self, generation):
		self.prev = generation
		self.generationCount += 1
	def StopCondition(self):
		return self.consecutiveIdleGenerationCount > max(self.maxIdleGenerations * 3 // 2, (task.clusterCount // 20 + 10))
	def ReproductionsCount(self):
		return 10 + task.clusterCount // 20 + self.generationCount // 5
	def NextGeneration(self, cur):
		def Reproduction(cur):
			for i in range(self.ReproductionsCount()):
				tour = copy.deepcopy(self.prev.tours[i])
				cur.Add(tour)
		def Crossover(cur):
			def CrossoverTries():
				return 8 * self.ReproductionsCount()
			for i in range(CrossoverTries()):
				tour = Tour(self.prev, crossoverStrategy)
				tour.Improve()
				if not cur.Contains(tour):
					cur.Add(tour)
		def Mutation(cur):
			def MutationTries():
				return 2 * self.ReproductionsCount()
			def mutationOperatorRun(source):
				values = source.vertices
				M = task.clusterCount

				rotation = random.randint(0, M - 1)
				values = Rotate(values, rotation)

				lens = int(M * random.uniform(0.05, 0.3))
				newPos = random.randint(0, M - lens - 1) + 1

				newValues = [0 for i in range(M)]
				for i in range(newPos):
					newValues[i] = values[lens + i]
				for i in range(lens):
					newValues[i + newPos] = values[i]
				for i in range(M - lens - newPos):
					newValues[i + newPos + lens] = values[i + newPos + lens]

				return Tour(None, vertices = newValues)

			for i in range(MutationTries()):
				index = mutationStrategy.Run(self.prev)

				tour = mutationOperatorRun(self.prev.tours[index])
				tour.Improve()
				if not cur.Contains(tour):
					cur.Add(tour)
		Reproduction(cur)
		Crossover(cur)
		Mutation(cur)
		cur.Sort()
		if cur.tours[0].length != self.prev.tours[0].length:
			self.maxIdleGenerations = max(self.maxIdleGenerations, self.consecutiveIdleGenerationCount)
			self.consecutiveIdleGenerationCount = 0
		else:
			self.consecutiveIdleGenerationCount += 1
	def run(self):
		generation = Generation(0)
		self.FirstGeneration(generation)
		self.AddGeneration(generation)
		while not self.StopCondition():
			generation = Generation(self.prev.generationIndex + 1)
			self.NextGeneration(generation)
			self.AddGeneration(generation)

def check_cluster():
	return len(task.clusters) <= 1024 and len(task.m) <= 1024 * 5

def optimal_solution(cluster, matrix):
	solver = Algorithm(matrix, cluster)
	solver.run()
	return solver.prev.tours[0].vertices