# ECG Beat Scrapper
R-R Interval 을 기점으로 비트를 잘라내는 프로젝트이다. 이론은 간단한데 코드로 작성하자니 어렵다.

## Logic
A Record가 있다고 하자. A 레코드는 길이가 3시간 분량이고 R-R Interval 길이가 다양하다고 가정하면 여기서 평균을 구한다. 각 평균을 구해 평균에 맞게 갖고오는데 만일 비트가 짧거나 맞지 않는다면 Zero-padding을 해주는 것이 이번 프로젝트(?) 의 목표이다.