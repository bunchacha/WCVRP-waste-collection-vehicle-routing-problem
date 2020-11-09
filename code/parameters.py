# -*- coding: utf-8 -*-

import math
import copy
import time
import random
from random import randint, shuffle, choice, random, uniform

class methodAndParameters:  # 입력 파라미터 설정 및 계산 관련 메소드들
    numGen = 50    # 해집단 수

    def setDistanceMatrix(self, distanceMatrix):    # distanceMatrix = 거리 행렬
        self.distanceMatrix = distanceMatrix

    def setTimeMatrix(self, timeMatrix):    # timeMatrix = 시간 행렬
        self.timeMatrix = timeMatrix

    def setProducts(self, products):    # products = 제품 테이블
        self.products = sorted(copy.deepcopy(products), key=lambda x: x[0])

    def setVehicles(self, vehicles):    # vehicles = 차량 테이블
        self.vehicles = sorted(copy.deepcopy(vehicles), key=lambda x: x[0])

    def getNumVehicles(self):  # numVehicles = 차량 수
        self.numVehicles = len(self.vehicles)

    def getNumProducts(self):  # numProducts = 제품 수
        self.numProducts = len(self.products)

    def getNumDepots(self):  # depots = 차고지 테이블, numDepots = 차고지 수, depotsVehicles = 차고지별 배정 차량 수
        self.depots = []
        self.depots.append(self.vehicles[0][0])
        for i in range(self.numVehicles-1):
            if self.vehicles[i][0] != self.vehicles[i + 1][0]:
                self.depots.append(self.vehicles[i + 1][0])
        self.numDepots = len(self.depots)
        self.depotsVehicles = []
        for i in range(self.numDepots):
            self.count = 0
            for j in range(self.numVehicles):
                if self.depots[i] == self.vehicles[j][0]:
                    self.count += 1
            self.depotsVehicles.append(self.count)

    def getDepotsProducts(self):    # depotsProducts = 차고지와 가까운 제품 테이블
        self.minDep = 0
        self.selectDep = 0
        self.depotsProducts = []
        for i in range(self.numDepots):
            self.depotsProducts.append([])

        for i in range(self.numProducts):
            for j in range(self.numDepots):
                if self.minDep == 0 or self.distanceMatrix[self.products[i][0]][self.depots[j]] < self.minDep:
                    self.minDep = self.distanceMatrix[self.products[i][0]][self.depots[j]]
                    self.selectDep = j
            self.depotsProducts[self.selectDep].append(self.products[i])

    def score(self, tour):  # 경로 거리 계산 함수
        self.distance = 0
        for i in range(len(tour) - 1):
            self.cur = tour[i]
            self.next_ = tour[i + 1]
            self.distance += self.distanceMatrix[self.cur][self.next_]
        return self.distance

    def timeScore(self, tour):  # 경로 시간 계산 함수
        self.time = 0
        for i in range(len(tour) - 1):
            self.cur = tour[i]
            self.next_ = tour[i + 1]
            self.time += self.timeMatrix[self.cur][self.next_]
        return self.time

    def avgScore(self, scored): # 50개 해 집단 평균 경로 거리 계산 함수
        # Find the average score of a population.
        self.total = 0
        for i in scored:
            self.total += i[1]
        self.total /= len(scored)
        return self.total

    def setCriteria(self, criteria):    # 차량이 특정 기준만큼 경로를 채웠을 경우 확률적으로 다음 차량으로 할당되게 하는 방법(기준)
        self.criteria = criteria

    def setObj(self, obj):  # 목적함수 값 설정
        self.obj = obj

class initialSolution:  # 50개의 초기 해(제품 순서) 결정을 위한 메소드들

    def initialization(self, inputParameter):   # 초기 해 알고리즘 실행 함수

        start_time = time.time()
        self.genRoute2 = []
        self.genRoute3 = []
        for i in range(inputParameter.numDepots):
            self.genRoute3.append([])
        self.numvi = 1

        for i in range(inputParameter.numDepots):
            self.cities = inputParameter.depotsProducts[i]
            # print("Vehcile #", str(self.numvi))
            self.mainOfInitial(inputParameter)  # 차고지 별 50개 해 생성 함수
            self.genRoute3[i].extend(self.genRoute2)
            self.genRoute2 = []
            self.numvi += 1

        # print("genRoute3")                  # genRoute3[차고지][0][염색체]
        # print(self.genRoute3)
        # print(len(self.genRoute3))          # 크기 genRoute3[0][0][0] + genRoute3[1][0][0] = 10
        self.genRoute4 = []
        for i in range(inputParameter.numGen):
            self.genRoute4.append([])

        for i in range(inputParameter.numGen):
            for j in range(inputParameter.numDepots):
                self.genRoute4[i].extend(self.genRoute3[j][0][i])

        # print("초기 해 생성 시간 : --- %s seconds ---" % (time.time() - start_time))

        # print("genRoute4")              # genRoute4[염색체]
        # print(self.genRoute4)
        # print(len(self.genRoute4[0]))
        # print(len(self.genRoute4))      # 크기 genRoute4 = 50 genRoute4[0] = 10

    def mainOfInitial(self, inputParameter):    # 차고지 별 50개 해 생성 함수
        self.nCandidates = inputParameter.numGen
        self.pop = self.generatePop2(self.nCandidates, inputParameter, self.cities) # 50개 해를 하나의 리스트에 표현
        self.genRoute2.append(self.pop)

    def productScore(self, inputParameter, tour):  # 경로 거리 계산 함수
        self.distance = 0
        for i in range(len(tour) - 1):
            self.cur = tour[i][0]
            self.next_ = tour[i + 1][0]
            self.distance += inputParameter.distanceMatrix[self.cur][self.next_]
        return self.distance

    def generateTour2(self, inputParameter, productsCopy):  # 차고지별 NNH 알고리즘 (1개 해)
        self.genRoute = []
        self.productsC = copy.deepcopy(productsCopy)
        if len(productsCopy) <= 1:
            return self.productsC
        self.start = randint(0, len(self.productsC) - 1)
        self.genRoute.append(self.productsC[self.start])
        self.productsC.pop(self.start)
        self.prev_value = 0
        while (len(self.genRoute) < len(productsCopy)):
            self.minA = 0
            self.selectA = 0
            self.prev_value = self.productScore(inputParameter, self.genRoute)  # 경로 거리 계산 함수
            for i in range(len(self.productsC)):
                self.genRoute.append(self.productsC[i])
                self.cur = self.genRoute[-2][0]
                self.next_ = self.genRoute[-1][0]
                self.score_check_val = self.prev_value + inputParameter.distanceMatrix[self.cur][self.next_]

                if self.minA == 0 or self.score_check_val < self.minA:
                    self.minA = self.score_check_val
                    self.selectA = i
                self.genRoute.pop()
            self.genRoute.append(self.productsC[self.selectA])
            self.productsC.pop(self.selectA)
        return self.genRoute

    def generatePop2(self, n, inputParameter, genRoute4):   # 50개 해를 하나의 리스트에 표현
        self.tour = []
        for i in range(n):
            self.tt = self.generateTour2(inputParameter, genRoute4) # 차고지 별 NNH 알고리즘 (1개 해)
            self.tour.append(copy.deepcopy(self.tt))
        return self.tour


class solution: # 50개의 최종 해(실제 경로) 중 가장 좋은 해 결정을 위한 메소드들
 
    def main(self, inputparameter, initial):    # 최종 해 알고리즘 실행 함수
        start_time = time.time()
        self.info = []
        self.info2 = []
        self.inFile = open("NEW_GA.txt", "w")
        self.paraK = 3  # 선택 연산 파라미터 값
        self.nCandidates = inputparameter.numGen
        self.mutateChance = 0.015
        self.parentPercent = 0.2  # 엘리트 보존 확률 20%
        self.pop = self.generatePop(self.nCandidates, inputparameter, initial.genRoute4)    # 50개 해를 하나의 리스트에 표현
        self.sumGen = 0
        self.iteration = 0
        self.vehiclesRoute = []     # 각 차량의 경로 # 차량 i 의 경로 = vehiclesRoute[i] = [방문하는 순서대로의 노드 index] = [int 형, int 형, ...]
        self.vehiclesDistance = []  # 각 차량의 운행 거리 # 차량 i 의 운행 거리 = vehiclesDistance[i] = float 형
        self.minDist = 0.0    # 모든 차량의 총 운행 거리 # totalVehiclesDistance = float 형
        self.vehiclesRoute_2 = []
        self.vehiclesDistance_2 = []
        self.vehiclesTime_2 = []
        self.vehiclesIndex_2 = []
        self.numCollectionProducts = 0
        self.numCollectionProductsRatio = 0
        self.vehiclesRoute_2_B = []
        self.vehiclesDistance_2_B = []
        self.vehiclesTime_2_B = []
        self.vehiclesIndex_2_B = []
        self.numCollectionProducts_B = 0
        self.numCollectionProductsRatio_B = 0
        self.timeVariance = 0.0
        self.timeSum = 0.0
        self.selectBalance = 0
        self.selectVariance = 0.0
        self.varianceSwitch = 0
        self.solutionDist = 0.0
        self.soultionTimeVariance = 0.0

        while (True):
            print("Generation #", self.iteration + 1)
            self.iteration += 1
            # Score the candidates
            self.scored = []

            for i in range(len(self.pop)):
                self.totaldistance = 0
                self.objTimeSum = 0.0
                self.objTimeAverage = 0.0
                self.objTimeVariance = 0.0
                self.maxTime = 0.0
                self.countVehicleNum = 0
                self.numCollectionProducts_B3 = 0
                for j in range(inputparameter.numVehicles):
                    self.totaldistance += inputparameter.score(self.pop[i][1][0][j])
                for k in range(inputparameter.numVehicles):
                    if self.pop[i][1][1][k] != 0:
                        self.countVehicleNum += 1
                        self.objTimeSum += self.pop[i][1][1][k]
                    if self.maxTime == 0.0 or self.pop[i][1][1][k] < self.maxTime:
                        self.maxTime = self.pop[i][1][1][k]

                self.objTimeAverage = float(self.objTimeSum) / self.countVehicleNum
                for l in range(inputparameter.numVehicles):
                    if self.pop[i][1][1][l] != 0:
                        self.objTimeVariance += pow(self.objTimeAverage - self.pop[i][1][1][l], 2)
                self.objTimeVariance = float(self.objTimeVariance) / self.countVehicleNum

                for m in range(inputparameter.numVehicles):
                    for n in range(len(self.pop[i][1][0][m])):
                        if self.pop[i][1][0][m][n] != inputparameter.vehicles[m][0] and self.pop[i][1][0][m][n] != inputparameter.vehicles[m][1]:
                            self.numCollectionProducts_B3 += 1

                self.objTimeVariance += (inputparameter.numProducts - self.numCollectionProducts_B3)*100000 # 제품 수거 패널티
                self.totaldistance += (inputparameter.numProducts - self.numCollectionProducts_B3)*100000   # 제품 수거 패널티
                self.scored.append([self.pop[i], self.totaldistance, self.objTimeVariance, self.maxTime])

            if inputparameter.obj == 0:  # 목적함수 : 차량 총 운행거리 최소화
                self.scored = sorted(self.scored, key=lambda x: x[1])
                self.selectParents = [x[0] for x in self.scored[:]]
                self.selectParentsScore = [x[1] for x in self.scored[:]]
                print(self.scored[0][1])

            elif inputparameter.obj == 1:  # 목적함수 : 차량 운행 시간 균등 분배
                self.scored = sorted(self.scored, key=lambda x: x[2])
                self.selectParents = [x[0] for x in self.scored[:]]
                self.selectParentsScore = [x[2] for x in self.scored[:]]

            else: # 목적함수 : 차량 운행 시간 균등 분배(max 값 최소화)
                self.scored = sorted(self.scored, key=lambda x: x[3])
                self.selectParents = [x[0] for x in self.scored[:]]
                self.selectParentsScore = [x[3] for x in self.scored[:]]

            self.fit = []   # 적합도 계산 (좋은 해가 부모 해로 선택될 확률이 높다)

            for i in range(len(self.pop)):
                self.fitness = ((self.selectParentsScore[-1] - self.selectParentsScore[i]) + (
                        self.selectParentsScore[-1] - self.selectParentsScore[0])) / self.paraK
                self.fit.append(self.fitness)
            self.sumOfFitnesses = 0

            for i in range(len(self.pop)):
                self.sumOfFitnesses += self.fit[i]

            self.parents = [x[0] for x in self.scored[:int(self.parentPercent * len(self.scored))]]
            self.newGen = self.parents[:]

            if inputparameter.obj == 0:  # 목적함수 : 차량 총 운행거리 최소화
                self.inFile.write("Generation #" + str(self.iteration) + ",    " + str(inputparameter.avgScore(self.scored)) + ",    "
                    + str(self.scored[-1][1]) + ",    " + str(self.scored[0][1]) + "\n")

            elif inputparameter.obj == 1:  # 목적함수 : 차량 운행 시간 균등 분배
                self.inFile.write("Generation #" + str(self.iteration) + ",    " + str(inputparameter.avgScore(self.scored)) + ",    "
                    + str(self.scored[-1][2]) + ",    " + str(self.scored[0][2]) + "\n")

            else:  # 목적함수 : 차량 운행 시간 균등 분배 (max 값 최소화)
                self.inFile.write("Generation #" + str(self.iteration) + ",    " + str(inputparameter.avgScore(self.scored)) + ",    "
                    + str(self.scored[-1][3]) + ",    " + str(self.scored[0][3]) + "\n")

            # self.sumtest = 0
            # for i in range(len(self.scored[0][0][1])):
            #     self.sumtest += len(self.scored[0][0][1][i])
            # print(self.sumtest)
            # print(self.sumGen)
            self.vehiclesRoute = self.scored[0][0][1][0]
            self.vehiclesDistance = [inputparameter.score(i) for i in self.vehiclesRoute]
            self.vehiclesTime = self.scored[0][0][1][1]
            self.vehiclesIndex = self.scored[0][0][1][2]
            self.solutionDist = self.scored[0][1]
            self.soultionTimeVariance = self.scored[0][2]

            # print("각 차량의 경로")
            # print(self.vehiclesRoute)
            # print("각 차량의 운행 거리")
            # print(self.vehiclesDistance)
            # print("모든 차량의 총 운행 거리")
            # print(self.minDist)
            # print("각 차량의 운행 시간")
            # print(self.vehiclesTime)
            # print("각 차량의 index")
            # print(self.vehiclesIndex)

            if inputparameter.obj == 0:  # 목적함수 : 차량 총 운행거리 최소화
                if self.minDist == self.scored[0][1]:
                    self.sumGen += 1
                else:
                    self.sumGen = 0
            elif inputparameter.obj == 1:  # 목적함수 : 차량 운행 시간 균등 분배
                if self.minDist == self.scored[0][2]:
                    self.sumGen += 1
                else:
                    self.sumGen = 0
            else:  # 목적함수 : 차량 운행 시간 균등 분배 (max 최소화)
                if self.minDist == self.scored[0][3]:
                    self.sumGen += 1
                else:
                    self.sumGen = 0

            if self.sumGen == 100:   # 최단 경로가 100회 이상 같을 경우 알고리즘 종료
                for i in range(inputparameter.numVehicles):
                    self.info.append([self.vehiclesRoute[i], self.vehiclesDistance[i], self.vehiclesTime[i], self.vehiclesIndex[i]])
                self.info = sorted(self.info, key=lambda x: x[3])
                for i in range(inputparameter.numVehicles):
                    self.vehiclesRoute_2.append(self.info[i][0])
                    self.vehiclesDistance_2.append(self.info[i][1])
                    self.vehiclesTime_2.append(self.info[i][2])
                    self.vehiclesIndex_2.append(self.info[i][3])
                for i in range(inputparameter.numVehicles):
                    for j in range(len(self.vehiclesRoute_2[i])):
                        if self.vehiclesRoute_2[i][j] != inputparameter.vehicles[i][0] and self.vehiclesRoute_2[i][j] != inputparameter.vehicles[i][1]:
                            self.numCollectionProducts += 1
                self.numCollectionProductsRatio = float(self.numCollectionProducts) / inputparameter.numProducts * 100.0
                self.countVehicles = 0
                for i in range(inputparameter.numGen):
                    for j in range(inputparameter.numVehicles):
                        if self.scored[i][0][1][1][j] != 0:
                            self.countVehicles += 1
                            self.timeSum += self.scored[i][0][1][1][j]
                    self.timeAverage = float(self.timeSum) / self.countVehicles
                    for j in range(inputparameter.numVehicles):
                        if self.scored[i][0][1][1][j] != 0.0:
                            self.timeVariance += pow(self.timeAverage - self.scored[i][0][1][1][j], 2)
                    self.timeVariance = float(self.timeVariance) / self.countVehicles
                    if self.varianceSwitch == 0:
                        self.selectBalance = i
                        self.selectVariance = self.timeVariance
                        self.varianceSwitch = 1
                    elif self.timeVariance < self.selectVariance:
                        self.selectBalance = i
                        self.selectVariance = self.timeVariance
                    self.timeVariance = 0.0
                    self.timeSum = 0.0
                    self.countVehicles = 0.0
                self.vehiclesRouteBalance = self.scored[self.selectBalance][0][1][0]
                self.vehiclesDistanceBalance = [inputparameter.score(i) for i in self.vehiclesRouteBalance]
                self.vehiclesTimeBalance = self.scored[self.selectBalance][0][1][1]
                self.vehiclesIndexBalance = self.scored[self.selectBalance][0][1][2]
                for i in range(inputparameter.numVehicles):
                    self.info2.append([self.vehiclesRouteBalance[i], self.vehiclesDistanceBalance[i], self.vehiclesTimeBalance[i], self.vehiclesIndexBalance[i]])
                self.info2 = sorted(self.info2, key=lambda x: x[3])
                for i in range(inputparameter.numVehicles):
                    self.vehiclesRoute_2_B.append(self.info2[i][0])
                    self.vehiclesDistance_2_B.append(self.info2[i][1])
                    self.vehiclesTime_2_B.append(self.info2[i][2])
                    self.vehiclesIndex_2_B.append(self.info2[i][3])
                for i in range(inputparameter.numVehicles):
                    for j in range(len(self.vehiclesRoute_2_B[i])):
                        if self.vehiclesRoute_2_B[i][j] != inputparameter.vehicles[i][0] and self.vehiclesRoute_2_B[i][j] != inputparameter.vehicles[i][1]:
                            self.numCollectionProducts_B += 1
                self.numCollectionProductsRatio_B = float(self.numCollectionProducts_B) / inputparameter.numProducts * 100.0
                self.minDist_B = self.scored[self.selectBalance][1]
                self.minVariance_B = self.scored[self.selectBalance][2]

                self.inFile.close()
                # print("최종 해 생성 시간 : --- %s seconds ---" % (time.time() - start_time))
                break
            if time.time() - start_time > 3600: # 알고리즘 실행 시간이 1시간을 초과할 경우 알고리즘 종료
                for i in range(inputparameter.numVehicles):
                    self.info.append([self.vehiclesRoute[i], self.vehiclesDistance[i], self.vehiclesTime[i], self.vehiclesIndex[i]])
                self.info = sorted(self.info, key=lambda x: x[3])
                for i in range(inputparameter.numVehicles):
                    self.vehiclesRoute_2.append(self.info[i][0])
                    self.vehiclesDistance_2.append(self.info[i][1])
                    self.vehiclesTime_2.append(self.info[i][2])
                    self.vehiclesIndex_2.append(self.info[i][3])
                for i in range(inputparameter.numVehicles):
                    for j in range(len(self.vehiclesRoute_2[i])):
                        if self.vehiclesRoute_2[i][j] != inputparameter.vehicles[i][0] and self.vehiclesRoute_2[i][j] != inputparameter.vehicles[i][1]:
                            self.numCollectionProducts += 1
                self.numCollectionProductsRatio = float(self.numCollectionProducts) / inputparameter.numProducts * 100.0
                self.countVehicles = 0
                for i in range(inputparameter.numGen):
                    for j in range(inputparameter.numVehicles):
                        if self.scored[i][0][1][1][j] != 0:
                            self.countVehicles += 1
                            self.timeSum += self.scored[i][0][1][1][j]
                    self.timeAverage = float(self.timeSum) / self.countVehicles
                    for j in range(inputparameter.numVehicles):
                        if self.scored[i][0][1][1][j] != 0.0:
                            self.timeVariance += pow(self.timeAverage - self.scored[i][0][1][1][j], 2)
                    self.timeVariance = float(self.timeVariance) / self.countVehicles
                    if self.varianceSwitch == 0:
                        self.selectBalance = i
                        self.selectVariance = self.timeVariance
                        self.varianceSwitch = 1
                    elif self.timeVariance < self.selectVariance:
                        self.selectBalance = i
                        self.selectVariance = self.timeVariance
                    self.timeVariance = 0.0
                    self.timeSum = 0.0
                    self.countVehicles = 0.0
                self.vehiclesRouteBalance = self.scored[self.selectBalance][0][1][0]
                self.vehiclesDistanceBalance = [inputparameter.score(i) for i in self.vehiclesRouteBalance]
                self.vehiclesTimeBalance = self.scored[self.selectBalance][0][1][1]
                self.vehiclesIndexBalance = self.scored[self.selectBalance][0][1][2]
                for i in range(inputparameter.numVehicles):
                    self.info2.append([self.vehiclesRouteBalance[i], self.vehiclesDistanceBalance[i], self.vehiclesTimeBalance[i], self.vehiclesIndexBalance[i]])
                self.info2 = sorted(self.info2, key=lambda x: x[3])
                for i in range(inputparameter.numVehicles):
                    self.vehiclesRoute_2_B.append(self.info2[i][0])
                    self.vehiclesDistance_2_B.append(self.info2[i][1])
                    self.vehiclesTime_2_B.append(self.info2[i][2])
                    self.vehiclesIndex_2_B.append(self.info2[i][3])
                for i in range(inputparameter.numVehicles):
                    for j in range(len(self.vehiclesRoute_2_B[i])):
                        if self.vehiclesRoute_2_B[i][j] != inputparameter.vehicles[i][0] and self.vehiclesRoute_2_B[i][j] != inputparameter.vehicles[i][1]:
                            self.numCollectionProducts_B += 1
                self.numCollectionProductsRatio_B = float(self.numCollectionProducts_B) / inputparameter.numProducts * 100.0
                self.minDist_B = self.scored[self.selectBalance][1]
                self.minVariance_B = self.scored[self.selectBalance][2]

                self.inFile.close()
                # print("최종 해 생성 시간 : --- %s seconds ---" % (time.time() - start_time))
                break

            while len(self.newGen) < self.nCandidates:  # 부모 해 선택
                self.point = uniform(0, self.sumOfFitnesses)
                self.sumF = 0
                self.selectParentsNumber1 = 0
                for i in range(len(self.pop)):
                    self.sumF += self.fit[i]
                    if (self.point <= self.sumF):
                        self.selectParentsNumber1 = i
                        break
                self.parent1 = self.selectParents[self.selectParentsNumber1]
                if self.iteration % 10 == 0:    # iteration 10의 배수 : 다른 차고지에 배정된 제품들간 연산
                    self.child1 = self.iterSwap10(inputparameter, self.parent1)
                elif self.iteration % 3 == 0:   # iteration 3의 배수 : 같은 차고지에 배정된 제품들간 연산
                    self.child1 = self.iterSwap3(inputparameter, self.parent1)
                else:   # iteration 1의 배수 : 같은 차고지의 같은 차량에 배정된 제품들간 연산
                    self.child1 = self.iterSwap(inputparameter, self.parent1)
                self.newGen.append(self.child1)

                """
                self.point2 = uniform(0, self.sumOfFitnesses)
                self.sumF2 = 0
                self.selectParentsNumber2 = 0
                for i in range(len(self.pop)):
                    self.sumF2 += self.fit[i]
                    if (self.point2 <= self.sumF2):
                        self.selectParentsNumber2 = i
                        break

                self.parent2 = self.selectParents[self.selectParentsNumber2]
                while self.iteration > 1 and self.parent1 == self.parent2:
                    self.point2 = uniform(0, self.sumOfFitnesses)
                    self.sumF2 = 0
                    for i in range(len(self.pop)):
                        self.sumF2 += self.fit[i]
                        if (self.point2 <= self.sumF2):
                            self.selectParentsNumber2 = i
                            break

                    self.parent2 = self.selectParents[self.selectParentsNumber2]
                self.child2, self.child3 = self.crossover(inputparameter, self.parent1, self.parent2)
                self.newGen.append(self.child2)
                self.newGen.append(self.child3)
                """
                self.pop = self.newGen[:int(len(self.scored))]

                if inputparameter.obj == 0:  # 목적함수 : 차량 총 운행거리 최소화
                    self.minDist = self.scored[0][1]
                elif inputparameter.obj == 1:  # 목적함수 : 차량 운행 시간 균등 분배
                    self.minDist = self.scored[0][2]
                else:  # 목적함수 : 차량 운행 시간 균등 분배 (max 최소화)
                    self.minDist = self.scored[0][3]


    def generateRoute(self, inputparameter, genRoute):  # 최종 해(실제 경로) 생성 함수 (1개 해)
        self.routeInfo = []
        for i in range(3):
            self.routeInfo.append([])
        self.route = []
        for i in range(inputparameter.numVehicles):
            self.route.append([])

        self.vehicles_2 = copy.deepcopy(inputparameter.vehicles)     # vehicles 투입 순서 결정
        self.vehicles_3 = copy.deepcopy(inputparameter.vehicles)
        # shuffle(self.vehicles_2)
        self.vehiclesIndex = []
        self.vehiclesCapacity = []
        for i in range(len(self.vehicles_3)):
            self.vehiclesCapacity.append(self.vehicles_3[i][2])
            self.vehiclesIndex.append(self.vehicles_3[i][6])

        self.numP = 0
        self.numV = 0
        self.numLastP = 0
        self.switchEssentialCheck = 0
        self.switchVehicle = 0
        self.vehiclesTime = []
        self.switchAllocation = 0

        self.genRoute_3 = copy.deepcopy(genRoute)

        for i in range(len(self.vehicles_3)):
            self.vehiclesTime.append(0.0)

        #self.switchManVehi = 1
        self.switchManVehi = randint(0, 1)

        if self.switchManVehi == 0:
            if self.switchEssentialCheck == 0:
                for i in range(inputparameter.numVehicles):
                    self.route[i].append(self.vehicles_3[i][0])
                    if self.vehicles_3[i][5] == 1:  # 필수 차량에 제품 1대씩 우선 배정
                        self.k = i
                        if len(self.genRoute_3) <= 1:
                            self.route[self.k].append(self.genRoute_3[0][0])
                            self.vehiclesTime[self.k] += self.genRoute_3[0][2]
                            self.vehiclesCapacity[self.k] = self.vehiclesCapacity[self.k] - self.genRoute_3[0][1]
                            self.genRoute_3.pop()
                        else:
                            self.minDep = 0
                            self.select = 0

                            for j in range(len(self.genRoute_3)):
                                if self.minDep == 0 or inputparameter.distanceMatrix[self.genRoute_3[j][0]][
                                    self.vehicles_3[i][0]] < self.minDep:
                                    self.minDep = inputparameter.distanceMatrix[self.genRoute_3[j][0]][
                                        self.vehicles_3[i][0]]
                                    self.select = j
                            self.route[self.k].append(self.genRoute_3[self.select][0])
                            self.vehiclesTime[self.k] += self.genRoute_3[self.select][2]
                            self.vehiclesCapacity[self.k] = self.vehiclesCapacity[self.k] - \
                                                            self.genRoute_3[self.select][1]
                            self.genRoute_3.pop(self.select)

                self.switchEssentialCheck = 1


        else:
            if self.switchEssentialCheck == 0:
                for i in range(inputparameter.numVehicles):
                    self.route[i].append(self.vehicles_3[i][0])
                    if self.vehicles_3[i][5] == 1:  # 필수 차량에 제품 1대씩 우선 배정
                        self.k = i
                        if len(self.genRoute_3) <= 1:
                            self.route[self.k].append(self.genRoute_3[0][0])
                            self.vehiclesTime[self.k] += self.genRoute_3[0][2]
                            self.vehiclesCapacity[self.k] = self.vehiclesCapacity[self.k] - self.genRoute_3[0][1]
                            self.genRoute_3.pop()
                        else:


                            self.h = randint(0, len(self.genRoute_3) - 1)
                            self.route[self.k].append(self.genRoute_3[self.h][0])
                            self.vehiclesTime[self.k] += self.genRoute_3[self.h][2]
                            self.vehiclesCapacity[self.k] = self.vehiclesCapacity[self.k] - self.genRoute_3[self.h][1]
                            self.genRoute_3.pop(self.h)

                self.switchEssentialCheck = 1

        if len(self.genRoute_3) == 0:
            for i in range(inputparameter.numVehicles):
                self.numV2 = i
                if len(self.route[self.numV2]) != 0 and self.route[self.numV2][-1] == self.vehicles_3[self.numV2][1]:
                    self.route[self.numV2].append(self.vehicles_3[self.numV2][0])
                elif len(self.route[self.numV2]) != 0 and self.route[self.numV2][-1] != self.vehicles_3[self.numV2][0]:
                    self.route[self.numV2].append(self.vehicles_3[self.numV2][1])
                    self.vehiclesTime[self.numV2] += self.vehicles_3[self.numV2][4]
                    self.route[self.numV2].append(self.vehicles_3[self.numV2][0])


        #while (self.numP < len(self.genRoute_3) and len(self.vehicles_2) > 0):
        while (self.numP < len(self.genRoute_3) and self.numV < inputparameter.numVehicles):

            if self.switchVehicle == 0:
                self.minDep = [0]
                self.select = 0
                for i in range(len(self.vehicles_2)):
                    for j in range(inputparameter.numVehicles):
                        if self.vehicles_3[j] == self.vehicles_2[i]:
                            self.selectJoin = j
                    if self.minDep == [0] or inputparameter.distanceMatrix[self.genRoute_3[self.numP][0]][self.route[self.selectJoin][-1]] < self.minDep[0]:
                        self.minDep = [inputparameter.distanceMatrix[self.genRoute_3[self.numP][0]][self.route[self.selectJoin][-1]]]
                        self.minDep.append(i)
                        self.select = i
                    elif inputparameter.distanceMatrix[self.genRoute_3[self.numP][0]][self.route[self.selectJoin][-1]] == self.minDep[0]:
                        self.minDep.append(i)
                        self.select = self.minDep[randint(1, len(self.minDep) - 1)]
                    """   
                    if self.minDep == [0] or inputparameter.distanceMatrix[self.genRoute_3[self.numP][0]][self.vehicles_2[i][0]] < self.minDep[0]:
                        self.minDep = [inputparameter.distanceMatrix[self.genRoute_3[self.numP][0]][self.vehicles_2[i][0]]]
                        self.minDep.append(i)
                        self.select = i
                    elif inputparameter.distanceMatrix[self.genRoute_3[self.numP][0]][self.vehicles_2[i][0]] == self.minDep[0]:
                        self.minDep.append(i)
                        self.select = self.minDep[randint(1, len(self.minDep) - 1)]
                    """
                self.selectCheck = 0
                for i in range(inputparameter.numVehicles):
                    if self.vehicles_3[i] == self.vehicles_2[self.select]:
                        self.selectCheck = i


            if len(self.route[self.selectCheck]) == 0 and self.switchVehicle == 0:
                self.route[self.selectCheck].append(self.vehicles_2[self.select][0])
                self.switchVehicle = 1

            if len(self.route[self.selectCheck]) != 0 and self.switchVehicle == 0:
                self.switchVehicle = 1

            self.route[self.selectCheck].append(self.genRoute_3[self.numP][0])
            self.vehiclesTime[self.selectCheck] += self.genRoute_3[self.numP][2]
            self.vehiclesCapacity[self.selectCheck] = self.vehiclesCapacity[self.selectCheck] - self.genRoute_3[self.numP][1]

            if self.vehiclesCapacity[self.selectCheck] > 0:  # 차량 용량 > 0
                self.route[self.selectCheck].append(self.vehicles_2[self.select][1])
                self.vehiclesTime[self.selectCheck] += self.vehicles_2[self.select][4]
                self.route[self.selectCheck].append(self.vehicles_2[self.select][0])
                if inputparameter.timeScore(self.route[self.selectCheck]) + self.vehiclesTime[self.selectCheck] <= self.vehicles_2[self.select][3]:  # 현재 차량 운행 시간 < 최대 차량 운행 시간
                    self.route[self.selectCheck].pop()
                    self.route[self.selectCheck].pop()
                    self.vehiclesTime[self.selectCheck] -= self.vehicles_2[self.select][4]
                    self.numP += 1

                    if self.switchAllocation == 0:
                        """
                        # case 1
                        if float((inputparameter.timeScore(self.route[self.selectCheck]) + self.vehiclesTime[self.selectCheck])) / self.vehicles_2[self.select][3] >= inputparameter.criteria:
                            self.yValue = (0.5 - 1) * (float((inputparameter.timeScore(self.route[self.selectCheck]) + self.vehiclesTime[self.selectCheck])) / self.vehicles_2[self.select][3] - 1) / (inputparameter.criteria - 1) + 1
                            self.randomValue = random() 
                            if self.randomValue <= self.yValue:
                                self.numV += 1
                                self.switchVehicle = 0
                                self.vehicles_2.pop(self.select)
                        """
                        """
                        # case 2
                        if float((inputparameter.timeScore(self.route[self.selectCheck]) + self.vehiclesTime[self.selectCheck])) / self.vehicles_2[self.select][3] >= inputparameter.criteria:
                            self.yValue = (-1) * (float((inputparameter.timeScore(self.route[self.selectCheck]) + self.vehiclesTime[self.selectCheck])) / self.vehicles_2[self.select][3] - 1) / (inputparameter.criteria - 1) + 1
                            self.randomValue = random() 
                            if self.randomValue <= self.yValue:
                                self.numV += 1
                                self.switchVehicle = 0
                                self.vehicles_2.pop(self.select)
                        """
                        """
                        # case 3
                        if float((inputparameter.timeScore(self.route[self.selectCheck]) + self.vehiclesTime[self.selectCheck])) / self.vehicles_2[self.select][3] >= inputparameter.criteria:
                            self.yValue = 0.5 * pow((float((inputparameter.timeScore(self.route[self.selectCheck]) + self.vehiclesTime[self.selectCheck])) / self.vehicles_2[self.select][3] - inputparameter.criteria), 2) / pow((1 - inputparameter.criteria), 2) + 0.5
                            self.randomValue = random()
                            if self.randomValue <= self.yValue:
                                self.numV += 1
                                self.switchVehicle = 0
                                self.vehicles_2.pop(self.select)
                        """
                        """
                        # case 4
                        if float((inputparameter.timeScore(self.route[self.selectCheck]) + self.vehiclesTime[self.selectCheck])) / self.vehicles_2[self.select][3] >= inputparameter.criteria:
                            self.yValue = 0.5 / (1 + math.exp(-50 * (float((inputparameter.timeScore(self.route[self.selectCheck]) + self.vehiclesTime[self.selectCheck])) / self.vehicles_2[self.select][3] - (inputparameter.criteria + (1 - inputparameter.criteria) / 2)))) + 0.5
                            self.randomValue = random()
                            if self.randomValue <= self.yValue:
                                self.numV += 1
                                self.switchVehicle = 0
                                self.vehicles_2.pop(self.select)
                        """
                        """
                        # case 5
                        self.yValue = 1 / (1 + math.exp(-30 * (float((inputparameter.timeScore(self.route[self.selectCheck]) + self.vehiclesTime[self.selectCheck])) / self.vehicles_2[self.select][3] - inputparameter.criteria)))
                        self.randomValue = random()
                        if self.randomValue <= self.yValue:
                            self.numV += 1
                            self.switchVehicle = 0
                            self.vehicles_2.pop(self.select)
                        """
                        """
                        # case 6   
                        self.yValue = pow(float((inputparameter.timeScore(self.route[self.selectCheck]) + self.vehiclesTime[self.selectCheck])) / self.vehicles_2[self.select][3], 2)
                        self.randomValue = random()
                        if self.randomValue <= self.yValue:
                            self.numV += 1
                            self.switchVehicle = 0
                            self.vehicles_2.pop(self.select)
                        """
                        """
                        # case 7
                        if float((inputparameter.timeScore(self.route[self.selectCheck]) + self.vehiclesTime[self.selectCheck])) / self.vehicles_2[self.select][3] >= inputparameter.criteria:
                            self.yValue = pow((float((inputparameter.timeScore(self.route[self.selectCheck]) +self.vehiclesTime[self.selectCheck])) /self.vehicles_2[self.select][3] - inputparameter.criteria), 2) / pow((1 - inputparameter.criteria), 2)
                            self.randomValue = random()
                            if self.randomValue <= self.yValue:
                                self.numV += 1
                                self.switchVehicle = 0
                                self.vehicles_2.pop(self.select)
                        """


                else:
                    self.route[self.selectCheck].pop()
                    self.route[self.selectCheck].pop()
                    self.vehiclesTime[self.selectCheck] -= self.vehicles_2[self.select][4]
                    self.route[self.selectCheck].pop()
                    self.vehiclesCapacity[self.selectCheck] += self.genRoute_3[self.numP][1]
                    self.vehiclesTime[self.selectCheck] -= self.genRoute_3[self.numP][2]
                    if self.route[self.selectCheck][-1] == self.vehicles_2[self.select][1]:
                        self.route[self.selectCheck].append(self.vehicles_2[self.select][0])
                    elif self.route[self.selectCheck][-1] != self.vehicles_2[self.select][0]:
                        self.route[self.selectCheck].append(self.vehicles_2[self.select][1])
                        self.vehiclesTime[self.selectCheck] += self.vehicles_2[self.select][4]
                        self.route[self.selectCheck].append(self.vehicles_2[self.select][0])
                    self.numV += 1
                    self.switchVehicle = 0
                    self.vehicles_2.pop(self.select)
                # print("1", self.route, self.numV, self.numP)

            elif self.vehiclesCapacity[self.selectCheck] == 0:  # 차량 용량 = 0
                self.route[self.selectCheck].append(self.vehicles_2[self.select][1])
                self.vehiclesTime[self.selectCheck] += self.vehicles_2[self.select][4]
                self.route[self.selectCheck].append(self.vehicles_2[self.select][0])

                if inputparameter.timeScore(self.route[self.selectCheck]) + self.vehiclesTime[self.selectCheck] <= self.vehicles_2[self.select][3]:  # 현재 차량 운행 시간 < 최대 차량 운행 시간
                    self.route[self.selectCheck].pop()
                    self.numP += 1
                    self.vehiclesCapacity[self.selectCheck] = self.vehicles_2[self.select][2]
                    #
                    # self.numV += 1
                    # self.switchVehicle = 0
                    # self.vehicles_2.pop(self.select)
                    #


                else:
                    self.route[self.selectCheck].pop()
                    self.route[self.selectCheck].pop()
                    self.route[self.selectCheck].pop()
                    self.vehiclesCapacity[self.selectCheck] += self.genRoute_3[self.numP][1]
                    self.vehiclesTime[self.selectCheck] -= self.genRoute_3[self.numP][2]
                    self.vehiclesTime[self.selectCheck] -= self.vehicles_2[self.select][4]
                    if self.route[self.selectCheck][-1] == self.vehicles_2[self.select][1]:
                        self.route[self.selectCheck].append(self.vehicles_2[self.select][0])
                    elif self.route[self.selectCheck][-1] != self.vehicles_2[self.select][0]:
                        self.route[self.selectCheck].append(self.vehicles_2[self.select][1])
                        self.vehiclesTime[self.selectCheck] += self.vehicles_2[self.select][4]
                        self.route[self.selectCheck].append(self.vehicles_2[self.select][0])
                    self.numV += 1
                    self.switchVehicle = 0
                    self.vehicles_2.pop(self.select)
                # print("2", self.route, self.numV, self.numP)

            else:  # 차량 용량 < 0
                self.route[self.selectCheck].pop()
                self.vehiclesCapacity[self.selectCheck] += self.genRoute_3[self.numP][1]
                self.vehiclesTime[self.selectCheck] -= self.genRoute_3[self.numP][2]
                self.route[self.selectCheck].append(self.vehicles_2[self.select][1])
                self.vehiclesTime[self.selectCheck] += self.vehicles_2[self.select][4]
                self.route[self.selectCheck].append(self.vehicles_2[self.select][0])

                if inputparameter.timeScore(self.route[self.selectCheck]) + self.vehiclesTime[self.selectCheck] <= self.vehicles_2[self.select][3]:  # 현재 차량 운행 시간 < 최대 차량 운행 시간
                    self.route[self.selectCheck].pop()
                    self.vehiclesCapacity[self.selectCheck] = self.vehicles_2[self.select][2]
                    #
                    # self.numV += 1
                    # self.switchVehicle = 0
                    # self.vehicles_2.pop(self.select)
                    #


                else:
                    self.route[self.selectCheck].pop()
                    self.route[self.selectCheck].pop()
                    self.vehiclesTime[self.selectCheck] -= self.vehicles_2[self.select][4]
                    if self.route[self.selectCheck][-1] == self.vehicles_2[self.select][1]:
                        self.route[self.selectCheck].append(self.vehicles_2[self.select][0])
                    elif self.route[self.selectCheck][-1] != self.vehicles_2[self.select][0]:
                        self.route[self.selectCheck].append(self.vehicles_2[self.select][1])
                        self.vehiclesTime[self.selectCheck] += self.vehicles_2[self.select][4]
                        self.route[self.selectCheck].append(self.vehicles_2[self.select][0])
                    self.vehiclesCapacity[self.selectCheck] = self.vehicles_2[self.select][2]
                    self.numV += 1
                    self.switchVehicle = 0
                    self.vehicles_2.pop(self.select)
                # print("3", self.route, self.numV, self.numP)

            if self.numP == len(self.genRoute_3):
                for i in range(inputparameter.numVehicles):
                    self.numV = i
                    if len(self.route[self.numV]) != 0 and self.route[self.numV][-1] == self.vehicles_3[self.numV][1]:
                        self.route[self.numV].append(self.vehicles_3[self.numV][0])
                    elif len(self.route[self.numV]) != 0 and self.route[self.numV][-1] != self.vehicles_3[self.numV][0]:
                        self.route[self.numV].append(self.vehicles_3[self.numV][1])
                        self.vehiclesTime[self.numV] += self.vehicles_3[self.numV][4]
                        self.route[self.numV].append(self.vehicles_3[self.numV][0])
                # print("4", self.route, self.numV, self.numP)

            if self.numV == inputparameter.numVehicles and self.switchAllocation == 1:
                break

            if self.numV == inputparameter.numVehicles and self.numP == self.numLastP:
                self.switchAllocation = 1

            if self.numP < len(self.genRoute_3) and self.numV == inputparameter.numVehicles:
                self.numV = 0
                self.switchVehicle = 0
                self.vehicles_2 = copy.deepcopy(inputparameter.vehicles)
                self.switchAllocation = 1
                self.numLastP = self.numP





            """
            if len(self.route[self.select]) == 0 and self.switchVehicle == 0:
                self.route[self.select].append(self.vehicles_2[self.select][0])
                self.switchVehicle = 1

            if len(self.route[self.select]) != 0 and self.switchVehicle == 0:
                self.switchVehicle = 1

            self.route[self.select].append(self.genRoute_3[self.numP][0])
            self.vehiclesTime[self.select] += self.genRoute_3[self.numP][2]
            self.vehiclesCapacity[self.select] = self.vehiclesCapacity[self.select] - self.genRoute_3[self.numP][1]

            if self.vehiclesCapacity[self.select] > 0:  # 차량 용량 > 0
                self.route[self.select].append(self.vehicles_2[self.select][1])
                self.vehiclesTime[self.select] += self.vehicles_2[self.select][4]
                self.route[self.select].append(self.vehicles_2[self.select][0])
                if inputparameter.timeScore(self.route[self.select]) + self.vehiclesTime[self.select] <= self.vehicles_2[self.select][3]:  # 현재 차량 운행 시간 < 최대 차량 운행 시간
                    self.route[self.select].pop()
                    self.route[self.select].pop()
                    self.vehiclesTime[self.select] -= self.vehicles_2[self.select][4]
                    self.numP += 1
                else:
                    self.route[self.select].pop()
                    self.route[self.select].pop()
                    self.vehiclesTime[self.select] -= self.vehicles_2[self.select][4]
                    self.route[self.select].pop()
                    self.vehiclesCapacity[self.select] += self.genRoute_3[self.numP][1]
                    self.vehiclesTime[self.select] -= self.genRoute_3[self.numP][2]
                    if self.route[self.select][-1] == self.vehicles_2[self.select][1]:
                        self.route[self.select].append(self.vehicles_2[self.select][0])
                    elif self.route[self.select][-1] != self.vehicles_2[self.select][0]:
                        self.route[self.select].append(self.vehicles_2[self.select][1])
                        self.vehiclesTime[self.select] += self.vehicles_2[self.select][4]
                        self.route[self.select].append(self.vehicles_2[self.select][0])
                    self.numV += 1
                    self.switchVehicle = 0
                    self.vehicles_2.pop(self.select)
                # print("1", self.route, self.numV, self.numP)

            elif self.vehiclesCapacity[self.select] == 0:  # 차량 용량 = 0
                self.route[self.select].append(self.vehicles_2[self.select][1])
                self.vehiclesTime[self.select] += self.vehicles_2[self.select][4]
                self.route[self.select].append(self.vehicles_2[self.select][0])
                if inputparameter.timeScore(self.route[self.select]) + self.vehiclesTime[self.select] <= self.vehicles_2[self.select][3]:  # 현재 차량 운행 시간 < 최대 차량 운행 시간
                    self.route[self.select].pop()
                    self.numP += 1
                    self.vehiclesCapacity[self.select] = self.vehicles_2[self.select][2]
                else:
                    self.route[self.select].pop()
                    self.route[self.select].pop()
                    self.route[self.select].pop()
                    self.vehiclesCapacity[self.select] += self.genRoute_3[self.numP][1]
                    self.vehiclesTime[self.select] -= self.genRoute_3[self.numP][2]
                    self.vehiclesTime[self.select] -= self.vehicles_2[self.select][4]
                    if self.route[self.select][-1] == self.vehicles_2[self.select][1]:
                        self.route[self.select].append(self.vehicles_2[self.select][0])
                    elif self.route[self.select][-1] != self.vehicles_2[self.select][0]:
                        self.route[self.select].append(self.vehicles_2[self.select][1])
                        self.vehiclesTime[self.select] += self.vehicles_2[self.select][4]
                        self.route[self.select].append(self.vehicles_2[self.select][0])
                    self.numV += 1
                    self.switchVehicle = 0
                    self.vehicles_2.pop(self.select)
                # print("2", self.route, self.numV, self.numP)

            else:  # 차량 용량 < 0
                self.route[self.select].pop()
                self.vehiclesCapacity[self.select] += self.genRoute_3[self.numP][1]
                self.vehiclesTime[self.select] -= self.genRoute_3[self.numP][2]
                self.route[self.select].append(self.vehicles_2[self.select][1])
                self.vehiclesTime[self.select] += self.vehicles_2[self.select][4]
                self.route[self.select].append(self.vehicles_2[self.select][0])
                if inputparameter.timeScore(self.route[self.select]) + self.vehiclesTime[self.select] <= self.vehicles_2[self.select][3]:  # 현재 차량 운행 시간 < 최대 차량 운행 시간
                    self.route[self.select].pop()
                    self.vehiclesCapacity[self.select] = self.vehicles_2[self.select][2]
                else:
                    self.route[self.select].pop()
                    self.route[self.select].pop()
                    self.vehiclesTime[self.select] -= self.vehicles_2[self.select][4]
                    if self.route[self.select][-1] == self.vehicles_2[self.select][1]:
                        self.route[self.select].append(self.vehicles_2[self.select][0])
                    elif self.route[self.select][-1] != self.vehicles_2[self.select][0]:
                        self.route[self.select].pop()
                        self.route[self.select].append(self.vehicles_2[self.select][1])
                        self.vehiclesTime[self.select] += self.vehicles_2[self.select][4]
                        self.route[self.select].append(self.vehicles_2[self.select][0])
                    self.numV += 1
                    self.switchVehicle = 0
                    self.vehicles_2.pop(self.select)
                # print("3", self.route, self.numV, self.numP)

            if self.numP == len(self.genRoute_3):
                for i in range(inputparameter.numVehicles):
                    self.numV = i
                    if len(self.route[self.numV]) != 0 and self.route[self.numV][-1] == self.vehicles_3[self.numV][1]:
                        self.route[self.numV].append(self.vehicles_3[self.numV][0])
                    elif len(self.route[self.numV]) != 0 and self.route[self.numV][-1] != self.vehicles_3[self.numV][0]:
                        self.route[self.numV].append(self.vehicles_3[self.numV][1])
                        self.vehiclesTime[self.numV] += self.vehicles_3[self.numV][4]
                        self.route[self.numV].append(self.vehicles_3[self.numV][0])
                # print("4", self.route, self.numV, self.numP)
        """

        for i in range(inputparameter.numVehicles):
            self.numV = i
            self.vehiclesTime[self.numV] += inputparameter.timeScore(self.route[self.numV])

        self.routeInfo[0] = copy.deepcopy(self.route)
        self.routeInfo[1] = copy.deepcopy(self.vehiclesTime)
        self.routeInfo[2] = copy.deepcopy(self.vehiclesIndex)

        return self.routeInfo

    def generateTour(self, inputparameter, gen):    # 제품 순서, 실제 경로를 하나의 리스트에 표현
        self.genRoute = copy.deepcopy(gen)
        self.solution = []
        for i in range(2):
            self.solution.append([])
        self.routeCalc = self.generateRoute(inputparameter, self.genRoute)  # 최종 해(실제 경로) 생성 함수
        self.solution[0] = copy.deepcopy(self.genRoute)
        self.solution[1] = copy.deepcopy(self.routeCalc)
        return self.solution

    def generatePop(self, n, inputparameter, genRoute4):    # 50개 해를 하나의 리스트에 표현
        # Generates a list of tours
        self.tour = []
        for i in range(n):
            self.tt = self.generateTour(inputparameter, genRoute4[i])   # 제품 순서, 실제 경로를 하나의 리스트에 표현
            self.tour.append(copy.deepcopy(self.tt))
        return self.tour

    def iterSwap10(self, inputparameter, tour): # iteration 10의 배수 : 다른 차고지에 배정된 제품들간 연산
        self.parentsGene = copy.deepcopy(tour[0])   # Swaps two random cities in a tour
        self.pos1 = randint(0, len(self.parentsGene) - 1)
        self.pos2 = randint(0, len(self.parentsGene) - 1)
        self.case1 = 0
        self.case2 = 0
        self.case3 = 0

        if self.parentsGene == []:
            self.case3 = 1
            return [self.parentsGene, self.generateRoute(inputparameter, self.parentsGene)]
        if len(self.parentsGene) == 1:
            self.case1 = 1
            self.pos1 = 0
            self.pos2 = 0
        if len(self.parentsGene) == 2:
            self.case2 = 1
            self.pos1 = 0
            self.pos2 = 1
        if self.case1 == 0 and self.case2 == 0 and self.case3 == 0:
            while self.pos1 >= self.pos2:
                self.pos1 = randint(0, len(self.parentsGene) - 1)
                self.pos2 = randint(0, len(self.parentsGene) - 1)
        self.offspring1 = copy.deepcopy(self.parentsGene)
        self.offspring2 = copy.deepcopy(self.parentsGene)
        self.offspring3 = copy.deepcopy(self.parentsGene)
        self.offspring4 = copy.deepcopy(self.parentsGene)
        self.offspring5 = copy.deepcopy(self.parentsGene)
        self.offspring1[self.pos1], self.offspring1[self.pos2] = self.offspring1[self.pos2], self.offspring1[self.pos1]
        self.offspring2[self.pos1], self.offspring2[self.pos2] = self.offspring2[self.pos2], self.offspring2[self.pos1]
        self.offspring3[self.pos1], self.offspring3[self.pos2] = self.offspring3[self.pos2], self.offspring3[self.pos1]
        self.offspring4[self.pos1], self.offspring4[self.pos2] = self.offspring4[self.pos2], self.offspring4[self.pos1]
        self.offspring5[self.pos1], self.offspring5[self.pos2] = self.offspring5[self.pos2], self.offspring5[self.pos1]

        if self.case1 == 1:
            self.offspring2 = self.offspring2
            self.offspring3 = self.offspring3
            self.offspring4 = self.offspring4
            self.offspring5 = self.offspring5
        elif self.case2 == 1:
            self.offspring2[self.pos1], self.offspring2[self.pos2] = self.offspring2[self.pos2], self.offspring2[self.pos1]
        elif self.pos1 != 0 and self.pos2 != len(self.parentsGene) - 1 and self.pos1 + 1 != self.pos2:
            self.offspring2[self.pos1 - 1], self.offspring2[self.pos1] = self.offspring2[self.pos1], self.offspring2[self.pos1 - 1]
            self.offspring3[self.pos1], self.offspring3[self.pos1 + 1] = self.offspring3[self.pos1 + 1], self.offspring3[self.pos1]
            self.offspring4[self.pos2 - 1], self.offspring4[self.pos2] = self.offspring4[self.pos2], self.offspring4[self.pos2 - 1]
            self.offspring5[self.pos2], self.offspring5[self.pos2 + 1] = self.offspring5[self.pos2 + 1], self.offspring5[self.pos2]
        elif self.pos1 == 0 and self.pos2 == len(self.parentsGene) - 1 and self.pos1 + 1 != self.pos2:
            self.offspring2[self.pos1], self.offspring2[self.pos1 + 1] = self.offspring2[self.pos1 + 1], self.offspring2[self.pos1]
            self.offspring3[self.pos2 - 1], self.offspring3[self.pos2] = self.offspring3[self.pos2], self.offspring3[self.pos2 - 1]
        elif self.pos1 == 0 and self.pos2 != len(self.parentsGene) - 1 and self.pos1 + 1 != self.pos2:
            self.offspring2[self.pos1], self.offspring2[self.pos1 + 1] = self.offspring2[self.pos1 + 1], self.offspring2[self.pos1]
            self.offspring3[self.pos2 - 1], self.offspring3[self.pos2] = self.offspring3[self.pos2], self.offspring3[self.pos2 - 1]
            self.offspring4[self.pos2], self.offspring4[self.pos2 + 1] = self.offspring4[self.pos2 + 1], self.offspring4[self.pos2]
        elif self.pos1 != 0 and self.pos2 == len(self.parentsGene) - 1 and self.pos1 + 1 != self.pos2:
            self.offspring2[self.pos1 - 1], self.offspring2[self.pos1] = self.offspring2[self.pos1], self.offspring2[self.pos1 - 1]
            self.offspring3[self.pos1], self.offspring3[self.pos1 + 1] = self.offspring3[self.pos1 + 1], self.offspring3[self.pos1]
            self.offspring4[self.pos2 - 1], self.offspring4[self.pos2] = self.offspring4[self.pos2], self.offspring4[self.pos2 - 1]
        elif self.pos1 != 0 and self.pos2 != len(self.parentsGene) - 1 and self.pos1 + 1 == self.pos2:
            self.offspring2[self.pos1 - 1], self.offspring2[self.pos1] = self.offspring2[self.pos1], self.offspring2[self.pos1 - 1]
            self.offspring3[self.pos1], self.offspring3[self.pos1 + 1] = self.offspring3[self.pos1 + 1], self.offspring3[self.pos1]
        elif self.pos1 == 0 and self.pos2 != len(self.parentsGene) - 1 and self.pos1 + 1 == self.pos2:
            self.offspring2[self.pos2], self.offspring2[self.pos2 + 1] = self.offspring2[self.pos2 + 1], self.offspring2[self.pos2]
        elif self.pos1 != 0 and self.pos2 == len(self.parentsGene) - 1 and self.pos1 + 1 == self.pos2:
            self.offspring2[self.pos1 - 1], self.offspring2[self.pos1] = self.offspring2[self.pos1], self.offspring2[self.pos1 - 1]
        self.offRoute1 = copy.deepcopy(self.generateRoute(inputparameter, self.offspring1))
        self.offRoute2 = copy.deepcopy(self.generateRoute(inputparameter, self.offspring2))
        self.offRoute3 = copy.deepcopy(self.generateRoute(inputparameter, self.offspring3))
        self.offRoute4 = copy.deepcopy(self.generateRoute(inputparameter, self.offspring4))
        self.offRoute5 = copy.deepcopy(self.generateRoute(inputparameter, self.offspring5))

        self.sumOff1 = 0
        self.sumOff2 = 0
        self.sumOff3 = 0
        self.sumOff4 = 0
        self.sumOff5 = 0
        for i in range(len(self.offRoute1[0])):
            self.sumOff1 += inputparameter.score(self.offRoute1[0][i])
        for i in range(len(self.offRoute2[0])):
            self.sumOff2 += inputparameter.score(self.offRoute2[0][i])
        for i in range(len(self.offRoute3[0])):
            self.sumOff3 += inputparameter.score(self.offRoute3[0][i])
        for i in range(len(self.offRoute4[0])):
            self.sumOff4 += inputparameter.score(self.offRoute4[0][i])
        for i in range(len(self.offRoute5[0])):
            self.sumOff5 += inputparameter.score(self.offRoute5[0][i])
        self.scoreOff = []
        self.scoreOff.append([self.offspring1, self.offRoute1, self.sumOff1])
        self.scoreOff.append([self.offspring2, self.offRoute2, self.sumOff2])
        self.scoreOff.append([self.offspring3, self.offRoute3, self.sumOff3])
        self.scoreOff.append([self.offspring4, self.offRoute4, self.sumOff4])
        self.scoreOff.append([self.offspring5, self.offRoute5, self.sumOff5])
        self.scoreOff = sorted(self.scoreOff, key=lambda x: x[2])

        return [self.scoreOff[0][0], self.scoreOff[0][1]]

    def iterSwap3(self, inputparameter, tour):  # iteration 3의 배수 : 같은 차고지에 배정된 제품들간 연산
        self.parentsGene = copy.deepcopy(tour[0])
        self.case1 = 0
        self.case2 = 0
        self.case3 = 0

        self.Nd = []
        for i in range(inputparameter.numDepots):
            self.Nd.append([])

        self.k = 0
        for i in range(inputparameter.numDepots):
            if i == inputparameter.numDepots - 1:
                self.Nd[i] = self.parentsGene[self.k:]
            else:
                self.Nd[i] = self.parentsGene[self.k:self.k + int(len(self.parentsGene) / inputparameter.numDepots)]
                self.k += int(len(self.parentsGene) / inputparameter.numDepots)
        self.pos11 = randint(0, inputparameter.numDepots - 1)

        if self.Nd[self.pos11] == []:
            self.case3 = 1
            return [self.parentsGene, self.generateRoute(inputparameter, self.parentsGene)]
        if len(self.Nd[self.pos11]) == 1:
            self.case1 = 1
            self.pos1 = 0
            self.pos2 = 0
        if len(self.Nd[self.pos11]) == 2:
            self.case2 = 1
            self.pos1 = 0
            self.pos2 = 1
        else:
            self.pos1 = randint(0, len(self.Nd[self.pos11]) - 1)   # Swaps two random cities in a tour
            self.pos2 = randint(0, len(self.Nd[self.pos11]) - 1)
        if self.case1 == 0 and self.case2 == 0 and self.case3 == 0:
            while self.pos1 >= self.pos2:
                self.pos1 = randint(0, len(self.Nd[self.pos11]) - 1)
                self.pos2 = randint(0, len(self.Nd[self.pos11]) - 1)
        self.offspring1 = copy.deepcopy(self.Nd[self.pos11])
        self.offspring2 = copy.deepcopy(self.Nd[self.pos11])
        self.offspring3 = copy.deepcopy(self.Nd[self.pos11])
        self.offspring4 = copy.deepcopy(self.Nd[self.pos11])
        self.offspring5 = copy.deepcopy(self.Nd[self.pos11])
        self.offspring1[self.pos1], self.offspring1[self.pos2] = self.offspring1[self.pos2], self.offspring1[self.pos1]
        self.offspring2[self.pos1], self.offspring2[self.pos2] = self.offspring2[self.pos2], self.offspring2[self.pos1]
        self.offspring3[self.pos1], self.offspring3[self.pos2] = self.offspring3[self.pos2], self.offspring3[self.pos1]
        self.offspring4[self.pos1], self.offspring4[self.pos2] = self.offspring4[self.pos2], self.offspring4[self.pos1]
        self.offspring5[self.pos1], self.offspring5[self.pos2] = self.offspring5[self.pos2], self.offspring5[self.pos1]

        if self.case1 == 1:
            self.offspring2 = self.offspring2
            self.offspring3 = self.offspring3
            self.offspring4 = self.offspring4
            self.offspring5 = self.offspring5
        elif self.case2 == 1:
            self.offspring2[self.pos1], self.offspring2[self.pos2] = self.offspring2[self.pos2], self.offspring2[self.pos1]
        elif self.pos1 != 0 and self.pos2 != len(self.Nd[self.pos11]) - 1 and self.pos1 + 1 != self.pos2:
            self.offspring2[self.pos1 - 1], self.offspring2[self.pos1] = self.offspring2[self.pos1], self.offspring2[self.pos1 - 1]
            self.offspring3[self.pos1], self.offspring3[self.pos1 + 1] = self.offspring3[self.pos1 + 1], self.offspring3[self.pos1]
            self.offspring4[self.pos2 - 1], self.offspring4[self.pos2] = self.offspring4[self.pos2], self.offspring4[self.pos2 - 1]
            self.offspring5[self.pos2], self.offspring5[self.pos2 + 1] = self.offspring5[self.pos2 + 1], self.offspring5[self.pos2]
        elif self.pos1 == 0 and self.pos2 == len(self.Nd[self.pos11]) - 1 and self.pos1 + 1 != self.pos2:
            self.offspring2[self.pos1], self.offspring2[self.pos1 + 1] = self.offspring2[self.pos1 + 1], self.offspring2[self.pos1]
            self.offspring3[self.pos2 - 1], self.offspring3[self.pos2] = self.offspring3[self.pos2], self.offspring3[self.pos2 - 1]
        elif self.pos1 == 0 and self.pos2 != len(self.Nd[self.pos11]) - 1 and self.pos1 + 1 != self.pos2:
            self.offspring2[self.pos1], self.offspring2[self.pos1 + 1] = self.offspring2[self.pos1 + 1], self.offspring2[self.pos1]
            self.offspring3[self.pos2 - 1], self.offspring3[self.pos2] = self.offspring3[self.pos2], self.offspring3[self.pos2 - 1]
            self.offspring4[self.pos2], self.offspring4[self.pos2 + 1] = self.offspring4[self.pos2 + 1], self.offspring4[self.pos2]
        elif self.pos1 != 0 and self.pos2 == len(self.Nd[self.pos11]) - 1 and self.pos1 + 1 != self.pos2:
            self.offspring2[self.pos1 - 1], self.offspring2[self.pos1] = self.offspring2[self.pos1], self.offspring2[self.pos1 - 1]
            self.offspring3[self.pos1], self.offspring3[self.pos1 + 1] = self.offspring3[self.pos1 + 1], self.offspring3[self.pos1]
            self.offspring4[self.pos2 - 1], self.offspring4[self.pos2] = self.offspring4[self.pos2], self.offspring4[self.pos2 - 1]
        elif self.pos1 != 0 and self.pos2 != len(self.Nd[self.pos11]) - 1 and self.pos1 + 1 == self.pos2:
            self.offspring2[self.pos1 - 1], self.offspring2[self.pos1] = self.offspring2[self.pos1], self.offspring2[self.pos1 - 1]
            self.offspring3[self.pos1], self.offspring3[self.pos1 + 1] = self.offspring3[self.pos1 + 1], self.offspring3[self.pos1]
        elif self.pos1 == 0 and self.pos2 != len(self.Nd[self.pos11]) - 1 and self.pos1 + 1 == self.pos2:
            self.offspring2[self.pos2], self.offspring2[self.pos2 + 1] = self.offspring2[self.pos2 + 1], self.offspring2[self.pos2]
        elif self.pos1 != 0 and self.pos2 == len(self.Nd[self.pos11]) - 1 and self.pos1 + 1 == self.pos2:
            self.offspring2[self.pos1 - 1], self.offspring2[self.pos1] = self.offspring2[self.pos1], self.offspring2[self.pos1 - 1]

        self.off1 = []
        self.off2 = []
        self.off3 = []
        self.off4 = []
        self.off5 = []

        self.Nd[self.pos11] = copy.deepcopy(self.offspring1)
        for i in range(inputparameter.numDepots):
            for k in range(len(self.Nd[i])):
                self.off1.append(self.Nd[i][k])
        self.offRoute1 = copy.deepcopy(self.generateRoute(inputparameter, self.off1))

        self.Nd[self.pos11] = copy.deepcopy(self.offspring2)
        for i in range(inputparameter.numDepots):
            for k in range(len(self.Nd[i])):
                self.off2.append(self.Nd[i][k])
        self.offRoute2 = copy.deepcopy(self.generateRoute(inputparameter, self.off2))

        self.Nd[self.pos11] = copy.deepcopy(self.offspring3)
        for i in range(inputparameter.numDepots):
            for k in range(len(self.Nd[i])):
                self.off3.append(self.Nd[i][k])
        self.offRoute3 = copy.deepcopy(self.generateRoute(inputparameter, self.off3))

        self.Nd[self.pos11] = copy.deepcopy(self.offspring4)
        for i in range(inputparameter.numDepots):
            for k in range(len(self.Nd[i])):
                self.off4.append(self.Nd[i][k])
        self.offRoute4 = copy.deepcopy(self.generateRoute(inputparameter, self.off4))

        self.Nd[self.pos11] = copy.deepcopy(self.offspring5)
        for i in range(inputparameter.numDepots):
            for k in range(len(self.Nd[i])):
                self.off5.append(self.Nd[i][k])
        self.offRoute5 = copy.deepcopy(self.generateRoute(inputparameter, self.off5))

        self.sumOff1 = 0
        self.sumOff2 = 0
        self.sumOff3 = 0
        self.sumOff4 = 0
        self.sumOff5 = 0
        for i in range(len(self.offRoute1[0])):
            self.sumOff1 += inputparameter.score(self.offRoute1[0][i])
        for i in range(len(self.offRoute2[0])):
            self.sumOff2 += inputparameter.score(self.offRoute2[0][i])
        for i in range(len(self.offRoute3[0])):
            self.sumOff3 += inputparameter.score(self.offRoute3[0][i])
        for i in range(len(self.offRoute4[0])):
            self.sumOff4 += inputparameter.score(self.offRoute4[0][i])
        for i in range(len(self.offRoute5[0])):
            self.sumOff5 += inputparameter.score(self.offRoute5[0][i])
        self.scoreOff = []
        self.scoreOff.append([self.off1, self.offRoute1, self.sumOff1])
        self.scoreOff.append([self.off2, self.offRoute2, self.sumOff2])
        self.scoreOff.append([self.off3, self.offRoute3, self.sumOff3])
        self.scoreOff.append([self.off4, self.offRoute4, self.sumOff4])
        self.scoreOff.append([self.off5, self.offRoute5, self.sumOff5])
        self.scoreOff = sorted(self.scoreOff, key=lambda x: x[2])

        return [self.scoreOff[0][0], self.scoreOff[0][1]]

    def iterSwap(self, inputparameter, tour):   # iteration 1의 배수 : 같은 차고지의 같은 차량에 배정된 제품들간 연산
        self.parentsGene = copy.deepcopy(tour[0])
        self.case1 = 0
        self.case2 = 0
        self.case3 = 0

        self.Nd = []
        for i in range(inputparameter.numDepots):
            self.Nd.append([])
            for j in range(inputparameter.depotsVehicles[i]):
                self.Nd[i].append([])

        self.k = 0
        for i in range(inputparameter.numDepots):
            for j in range(inputparameter.depotsVehicles[i]):
                if i == inputparameter.numDepots - 1 and j == inputparameter.depotsVehicles[i] - 1:
                    self.Nd[i][j] = self.parentsGene[self.k:]
                else:
                    self.Nd[i][j] = self.parentsGene[self.k:self.k + int(len(self.parentsGene) / inputparameter.numVehicles)]
                    self.k += int(len(self.parentsGene) / inputparameter.numVehicles)
        self.pos11 = randint(0, inputparameter.numDepots - 1)           # 여기서 에러
        self.pos22 = randint(0, inputparameter.depotsVehicles[self.pos11] - 1)

        if self.Nd[self.pos11][self.pos22] == []:
            self.case3 = 1
            return [self.parentsGene, self.generateRoute(inputparameter, self.parentsGene)]
        elif len(self.Nd[self.pos11][self.pos22]) == 1:
            self.case1 = 1
            self.pos1 = 0
            self.pos2 = 0
        elif len(self.Nd[self.pos11][self.pos22]) == 2:
            self.case2 = 1
            self.pos1 = 0
            self.pos2 = 1
        else:
            self.pos1 = randint(0, len(self.Nd[self.pos11][self.pos22]) - 1)    # Swaps two random cities in a tour
            self.pos2 = randint(0, len(self.Nd[self.pos11][self.pos22]) - 1)
        """
        print("Nd")
        print(self.Nd[self.pos11][self.pos22])
        print("pos11")
        print(self.pos11)
        print("pos22")
        print(self.pos22)
        print("pos1")
        print(self.pos1)
        print("pos2")
        print(self.pos2)
        """
        if self.case1 == 0 and self.case2 == 0 and self.case3 == 0:
            while self.pos1 >= self.pos2:
                self.pos1 = randint(0, len(self.Nd[self.pos11][self.pos22]) - 1)
                self.pos2 = randint(0, len(self.Nd[self.pos11][self.pos22]) - 1)

        self.offspring1 = copy.deepcopy(self.Nd[self.pos11][self.pos22])
        self.offspring2 = copy.deepcopy(self.Nd[self.pos11][self.pos22])
        self.offspring3 = copy.deepcopy(self.Nd[self.pos11][self.pos22])
        self.offspring4 = copy.deepcopy(self.Nd[self.pos11][self.pos22])
        self.offspring5 = copy.deepcopy(self.Nd[self.pos11][self.pos22])
        self.offspring1[self.pos1], self.offspring1[self.pos2] = self.offspring1[self.pos2], self.offspring1[self.pos1]
        self.offspring2[self.pos1], self.offspring2[self.pos2] = self.offspring2[self.pos2], self.offspring2[self.pos1]
        self.offspring3[self.pos1], self.offspring3[self.pos2] = self.offspring3[self.pos2], self.offspring3[self.pos1]
        self.offspring4[self.pos1], self.offspring4[self.pos2] = self.offspring4[self.pos2], self.offspring4[self.pos1]
        self.offspring5[self.pos1], self.offspring5[self.pos2] = self.offspring5[self.pos2], self.offspring5[self.pos1]

        if self.case1 == 1:
            self.offspring2 = self.offspring2
            self.offspring3 = self.offspring3
            self.offspring4 = self.offspring4
            self.offspring5 = self.offspring5
        elif self.case2 == 1:
            self.offspring2[self.pos1], self.offspring2[self.pos2] = self.offspring2[self.pos2], self.offspring2[self.pos1]
        elif self.pos1 != 0 and self.pos2 != len(self.Nd[self.pos11][self.pos22]) - 1 and self.pos1 + 1 != self.pos2:
            self.offspring2[self.pos1 - 1], self.offspring2[self.pos1] = self.offspring2[self.pos1], self.offspring2[self.pos1 - 1]
            self.offspring3[self.pos1], self.offspring3[self.pos1 + 1] = self.offspring3[self.pos1 + 1], self.offspring3[self.pos1]
            self.offspring4[self.pos2 - 1], self.offspring4[self.pos2] = self.offspring4[self.pos2], self.offspring4[self.pos2 - 1]
            self.offspring5[self.pos2], self.offspring5[self.pos2 + 1] = self.offspring5[self.pos2 + 1], self.offspring5[self.pos2]
        elif self.pos1 == 0 and self.pos2 == len(self.Nd[self.pos11][self.pos22]) - 1 and self.pos1 + 1 != self.pos2:
            self.offspring2[self.pos1], self.offspring2[self.pos1 + 1] = self.offspring2[self.pos1 + 1], self.offspring2[self.pos1]
            self.offspring3[self.pos2 - 1], self.offspring3[self.pos2] = self.offspring3[self.pos2], self.offspring3[self.pos2 - 1]
        elif self.pos1 == 0 and self.pos2 != len(self.Nd[self.pos11][self.pos22]) - 1 and self.pos1 + 1 != self.pos2:
            self.offspring2[self.pos1], self.offspring2[self.pos1 + 1] = self.offspring2[self.pos1 + 1], self.offspring2[self.pos1]
            self.offspring3[self.pos2 - 1], self.offspring3[self.pos2] = self.offspring3[self.pos2], self.offspring3[self.pos2 - 1]
            self.offspring4[self.pos2], self.offspring4[self.pos2 + 1] = self.offspring4[self.pos2 + 1], self.offspring4[self.pos2]
        elif self.pos1 != 0 and self.pos2 == len(self.Nd[self.pos11][self.pos22]) - 1 and self.pos1 + 1 != self.pos2:
            self.offspring2[self.pos1 - 1], self.offspring2[self.pos1] = self.offspring2[self.pos1], self.offspring2[self.pos1 - 1]
            self.offspring3[self.pos1], self.offspring3[self.pos1 + 1] = self.offspring3[self.pos1 + 1], self.offspring3[self.pos1]
            self.offspring4[self.pos2 - 1], self.offspring4[self.pos2] = self.offspring4[self.pos2], self.offspring4[self.pos2 - 1]
        elif self.pos1 != 0 and self.pos2 != len(self.Nd[self.pos11][self.pos22]) - 1 and self.pos1 + 1 == self.pos2:
            self.offspring2[self.pos1 - 1], self.offspring2[self.pos1] = self.offspring2[self.pos1], self.offspring2[self.pos1 - 1]
            self.offspring3[self.pos1], self.offspring3[self.pos1 + 1] = self.offspring3[self.pos1 + 1], self.offspring3[self.pos1]
        elif self.pos1 == 0 and self.pos2 != len(self.Nd[self.pos11][self.pos22]) - 1 and self.pos1 + 1 == self.pos2:
            self.offspring2[self.pos2], self.offspring2[self.pos2 + 1] = self.offspring2[self.pos2 + 1], self.offspring2[self.pos2]
        elif self.pos1 != 0 and self.pos2 == len(self.Nd[self.pos11][self.pos22]) - 1 and self.pos1 + 1 == self.pos2:
            self.offspring2[self.pos1 - 1], self.offspring2[self.pos1] = self.offspring2[self.pos1], self.offspring2[self.pos1 - 1]

        self.off1 = []
        self.off2 = []
        self.off3 = []
        self.off4 = []
        self.off5 = []

        self.Nd[self.pos11][self.pos22] = copy.deepcopy(self.offspring1)
        for i in range(inputparameter.numDepots):
            for j in range(inputparameter.depotsVehicles[i]):
                for k in range(len(self.Nd[i][j])):
                    self.off1.append(self.Nd[i][j][k])
        self.offRoute1 = copy.deepcopy(self.generateRoute(inputparameter, self.off1))

        self.Nd[self.pos11][self.pos22] = copy.deepcopy(self.offspring2)
        for i in range(inputparameter.numDepots):
            for j in range(inputparameter.depotsVehicles[i]):
                for k in range(len(self.Nd[i][j])):
                    self.off2.append(self.Nd[i][j][k])
        self.offRoute2 = copy.deepcopy(self.generateRoute(inputparameter, self.off2))

        self.Nd[self.pos11][self.pos22] = copy.deepcopy(self.offspring3)
        for i in range(inputparameter.numDepots):
            for j in range(inputparameter.depotsVehicles[i]):
                for k in range(len(self.Nd[i][j])):
                    self.off3.append(self.Nd[i][j][k])
        self.offRoute3 = copy.deepcopy(self.generateRoute(inputparameter, self.off3))

        self.Nd[self.pos11][self.pos22] = copy.deepcopy(self.offspring4)
        for i in range(inputparameter.numDepots):
            for j in range(inputparameter.depotsVehicles[i]):
                for k in range(len(self.Nd[i][j])):
                    self.off4.append(self.Nd[i][j][k])
        self.offRoute4 = copy.deepcopy(self.generateRoute(inputparameter, self.off4))

        self.Nd[self.pos11][self.pos22] = copy.deepcopy(self.offspring5)
        for i in range(inputparameter.numDepots):
            for j in range(inputparameter.depotsVehicles[i]):
                for k in range(len(self.Nd[i][j])):
                    self.off5.append(self.Nd[i][j][k])
        self.offRoute5 = copy.deepcopy(self.generateRoute(inputparameter, self.off5))

        self.sumOff1 = 0
        self.sumOff2 = 0
        self.sumOff3 = 0
        self.sumOff4 = 0
        self.sumOff5 = 0
        for i in range(len(self.offRoute1[0])):
            self.sumOff1 += inputparameter.score(self.offRoute1[0][i])
        for i in range(len(self.offRoute2[0])):
            self.sumOff2 += inputparameter.score(self.offRoute2[0][i])
        for i in range(len(self.offRoute3[0])):
            self.sumOff3 += inputparameter.score(self.offRoute3[0][i])
        for i in range(len(self.offRoute4[0])):
            self.sumOff4 += inputparameter.score(self.offRoute4[0][i])
        for i in range(len(self.offRoute5[0])):
            self.sumOff5 += inputparameter.score(self.offRoute5[0][i])
        self.scoreOff = []
        self.scoreOff.append([self.off1, self.offRoute1, self.sumOff1])
        self.scoreOff.append([self.off2, self.offRoute2, self.sumOff2])
        self.scoreOff.append([self.off3, self.offRoute3, self.sumOff3])
        self.scoreOff.append([self.off4, self.offRoute4, self.sumOff4])
        self.scoreOff.append([self.off5, self.offRoute5, self.sumOff5])
        self.scoreOff = sorted(self.scoreOff, key=lambda x: x[2])

        return [self.scoreOff[0][0], self.scoreOff[0][1]]


    def crossover(self, inputparameter, tour1, tour2):
        self.parents1CrossGene = copy.deepcopy(tour1[0])
        self.parents2CrossGene = copy.deepcopy(tour2[0])

        self.overlapData = []
        self.length = len(self.parents1CrossGene)
        for x in range(0, self.length - 1):
            for y in range(x + 1, self.length):
                if self.parents1CrossGene[x] == self.parents1CrossGene[y]:
                    self.overlapData.append(self.parents1CrossGene[x])
                    break
        # Takes the first half of each tour and uses those to create two new tours
        self.n = len(self.parents1CrossGene) // 2
        self.tour3 = []
        self.tour4 = []
        for i in range(self.n):
            self.tour3.append(self.parents1CrossGene[i])
            self.tour4.append(self.parents2CrossGene[i])

        for i in self.parents1CrossGene:
            if i not in self.tour4:
                self.tour4.append(i)
        for i in self.parents2CrossGene:
            if i not in self.tour3:
                self.tour3.append(i)
        for i in self.overlapData:
            while (self.tour3.count(i) < self.parents1CrossGene.count(i)):
                self.tour3.append(i)
        for i in self.overlapData:
            while (self.tour4.count(i) < self.parents2CrossGene.count(i)):
                self.tour4.append(i)

        self.crossRoute1 = copy.deepcopy(self.generateRoute(inputparameter, self.tour3))
        self.crossRoute2 = copy.deepcopy(self.generateRoute(inputparameter, self.tour4))
        self.crossSolution1 = [self.tour3, self.crossRoute1]
        self.crossSolution2 = [self.tour4, self.crossRoute2]

        return self.crossSolution1, self.crossSolution2


        """
        def mutate(tour):
            # Swaps two random cities in a tour
            pos1 = randint(0, len(tour) - 1)
            pos2 = randint(0, len(tour) - 1)
            while pos1 == pos2:
                pos2 = randint(0, len(tour) - 1)
            tour[pos1], tour[pos2] = tour[pos2], tour[pos1]
        """