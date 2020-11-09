# -*- coding: utf-8 -*-

import codecs
import csv
import time
import copy
import math
import random

start_time = time.time()

# 입력 데이터 형식

# 1. 거리 행렬

# (14*14 행렬) 제품 10개(index 0~9), 차고지 2개(index 10~11), 매립지 2개(index 12~13)
# 노드 i, 노드 j 사이의 거리 = distanceMatrix[i][j] = float 형
"""
distanceMatrix = []

f = codecs.open('distance.csv', 'r', encoding="utf-8-sig")
csvReader = csv.reader(f)
for row in csvReader:
    tempMat = []
    ia = iter(row)
    for i in ia:
        tempMat.append(float(i))
    distanceMatrix.append(tempMat)
f.close()


distanceMatrix = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], [1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], [4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0], [11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0], [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0], [13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]]


print("거리 행렬")
print(distanceMatrix)


# 2. 제품 테이블

# 제품 10개(0~9)
# 제품 i 의 정보 = product[i] = [제품 index, 쓰레기양(혹은 쓰레기통 용량)] = [int 형, float 형]
product = []


product = [[0, 10.0], [1, 10.0], [2, 10.0], [3, 10.0], [4, 10.0], [5, 20.0], [6, 20.0], [7, 20.0], [8, 20.0], [9, 20.0]]

print("제품 테이블")
print(product)
print("제품 개수")
print(len(product))

# 3. 차량 테이블

# 차량 4대
# 차량 i 의 정보 = vehicles[i] = [차고지 index, 매립지 index, 용량, 가용 시간] = [int 형, int 형, float 형, float 형]
vehicles = []


vehicles = [[10, 12, 40.0, 480.0], [10, 12, 50.0, 360.0], [11, 13, 40.0, 480.0], [11, 13, 50.0, 360.0]]
vehicles = [[11, 13, 40.0, 480.0], [11, 13, 50.0, 360.0], [10, 12, 40.0, 480.0], [10, 12, 50.0, 360.0]]

print("차량 테이블")
print(vehicles)
print("차량 대수")
print(len(vehicles))



depotsVehicles = sorted(copy.deepcopy(vehicles), key=lambda x: x[0])

depots = []
depots.append(depotsVehicles[0][0])
for i in range(4-1):
    if depotsVehicles[i][0] != depotsVehicles[i+1][0]:
        depots.append(depotsVehicles[i+1][0])
print(depots)

# 출력 데이터

# 1. 각 차량의 경로

# 차량 i 의 경로 = vehiclesRoute[i] = [방문하는 순서대로의 노드 index] = [int 형, int 형, ...]
vehiclesRoute = []

vehiclesRoute = [[vehicles[0][0], product[0][0], product[1][0], product[8][0], vehicles[0][1], vehicles[0][0]], [vehicles[1][0], product[2][0], product[3][0], product[9][0], vehicles[1][1], vehicles[1][0]], [vehicles[2][0], product[4][0], product[5][0], vehicles[2][1], vehicles[2][0]], [vehicles[3][0], product[6][0], product[7][0], vehicles[3][1], vehicles[3][0]]]
print("각 차량의 경로")
print(vehiclesRoute)


# 2. 각 차량의 운행 거리

# 차량 i 의 운행 거리 = vehiclesDistance[i] = float 형
vehiclesDistance = []
def score(tour):
    # Traverses the cities in order and finds the total distance.
    distance = 0
    for i in range(len(tour)-1):
        cur = tour[i]
        next = tour[i + 1]
        distance += distanceMatrix[cur][next]
    return distance
vehiclesDistance = [score(i) for i in vehiclesRoute]

vehiclesDistance = [24.0, 20.0, 18.0, 14.0]
print("각 차량의 운행 거리")
print(vehiclesDistance)


# 3. 모든 차량의 총 운행 거리

# 모든 차량의 운행 거리 = totalVehiclesDistance = float 형
totalVehiclesDistance = 0.0
for i in vehiclesDistance: 
    totalVehiclesDistance += i

totalVehiclesDistance = 76.0
print("모든 차량의 총 운행 거리")
print(totalVehiclesDistance)
"""

inFile = open("NEWlarge5.txt", "w")

numDepots = 5
numProducts = 400
numLandfills = 5
numVehicles = 20
numDepotsVehicles = 4
numNodes = numDepots + numProducts + numLandfills
x = []
y = []
nodes = []



for i in range(numNodes):
    x.append(random.randint(0, 170))
    y.append(random.randint(0, 170))

for i in range(numNodes):
    nodes.append((x[i], y[i]))

products = nodes[:numProducts]
depots = nodes[numProducts:numProducts+numDepots]
landfills = nodes[numProducts+numDepots:]
"""
#2/10/4
depots = nodes[:numDepots]
landfills = nodes[numDepots:numDepots+numLandfills]
products = nodes[numDepots+numLandfills:]
"""
print(nodes)
print(products)
print(depots)
print(landfills)

def calcDistance(x, y):
    return math.sqrt(math.pow(x[0]-y[0], 2)+math.pow(x[1]-y[1], 2))

distanceMatrix = []


for i in range(len(nodes)):
    distanceMatrix.append([])
    for j in range(len(nodes)):
        distanceMatrix[i].append(calcDistance(nodes[i], nodes[j]))
    print(len(distanceMatrix[i]))

for i in range(len(nodes)):
    for j in range(len(nodes)):
        inFile.write(str(calcDistance(nodes[i], nodes[j])) + ",")
    inFile.write("\n")

inFile.close()




products = []


for i in range(numProducts):
    products.append([])
    products[i].append(i)
    products[i].append(10)
    products[i].append(0)

print(products)
print(len(products))


"""

vehicles = []



mandatory = 1
numProducts2 = numProducts
numDepots2 = numDepots
j = 0
k = 0
for i in range(numVehicles):
    vehicles.append([])
    if mandatory == 0:
        vehicles[i].append(numProducts2 + k)
        vehicles[i].append(numProducts2 + numDepots2 + k)
        vehicles[i].append(200)
        vehicles[i].append(480)
        vehicles[i].append(0)
        vehicles[i].append(0)
        vehicles[i].append(i)
        mandatory = 1
    else:
        vehicles[i].append(numProducts2 + k)
        vehicles[i].append(numProducts2 + numDepots2 + k)
        vehicles[i].append(200)
        vehicles[i].append(480)
        vehicles[i].append(0)
        vehicles[i].append(1)
        vehicles[i].append(i)
        mandatory = 0

    if j < numDepotsVehicles-1:
        j += 1
    else:
        j = 0
        k += 1

print(vehicles)
print(len(vehicles))


inFile.write(str(vehicles))
inFile.close()
"""