## 프로젝트 소개
이 프로젝트는 **SLAM 기반 로봇 주행**과 **환경 장애물 인식**, **미지 영역 맵핑 및 객체 탐색**을 목표로 진행되었습니다. TurtleBot을 활용하여 주어진 공간을 스스로 탐색하고, 목표 객체를 인식하여 맵 좌표계에 표현하는 알고리즘을 구현

## explore_node.py
- SLAM toolbox를 활용해서 매핑을 진행하였고, NAV2를 활용해서 미지의 영역을 탐색하는 알고리즘을 구성함
- 미지의 영역과 탐색된 영역의 경계선으로 이동하도록 함
- 미탐색 영역이 일정 수치 이하로 내려가면 탐색을 마치도록 함

## feature_matching.py
- ext_orig, man_orig 두 사진으로 feature matching을 수행함
- 맵 탐색 도중 두 사진을 발견하게되면 해당 위치를 map에 marking함

## 미탐색 영역 자동 매핑 영상

[![Video](https://img.youtube.com/vi/DobwPF-5XSg/0.jpg)](https://www.youtube.com/watch?v=DobwPF-5XSg)
