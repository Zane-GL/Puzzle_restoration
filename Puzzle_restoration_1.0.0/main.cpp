#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <string>
#include <cstdlib>
#include <utility>
#include <opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;

/*
 * 问题: 将4张打乱顺序的碎片复原，将复原好的图片展示出来
 * 思路: 1. 将4张碎片的左右边缘提取保存
 *       2. 左右边缘两两对比，将相似度超过预设阈值的碎片执行拼接操作，得到左右拼接好的碎片
 *       3. 提取左右拼接好的碎片的上下边缘
 *       4. 上下边缘两两对比，将相似度超过预设阈值的碎片执行拼接操作，得到原图
*/

int n = 0;//左右拼接函数迭代器
int m = 0;//上下拼接函数迭代器

//读取碎片
vector<Mat> fragments_Imread(string files_name);
vector<Mat> fragments_LR_Imread(string files_name);     //读取左右拼接好的碎片

//保存每张碎片的左右边缘
vector <vector<Mat>> edge_resection_LR(const vector <Mat>& fragments);

//直方图对比
bool compare_by_hist(const Mat& img1, const Mat& img2);

//左右拼接
void picture_stitching_LR(const Mat& img1, const Mat& img2);

//对每张碎片的左右边缘相互对比拼接
void alignment_and_splicing_LR(const vector <Mat>& fragments, const vector<vector<Mat>>& resection_LR);//参数：碎片；碎片的左右边缘

//保存每张碎片的上下边缘
vector <vector<Mat>> edge_resection_TB(const vector <Mat>& fragments_LR);

//上下拼接
void picture_stitching_TB(const Mat& img1, const Mat& img2);

//对左右拼接好的碎片进行上下对比拼接
void alignment_and_splicing_TB(const vector <Mat>& fragments_LR, const vector<vector<Mat>>& resection_TB);


int main() {
    vector<Mat> fragments = fragments_Imread("res/fragments/");              //读取碎片

    vector<vector<Mat> > resection_LR = edge_resection_LR(fragments);                   //保存每张碎片的左右边缘

    alignment_and_splicing_LR(fragments,resection_LR);                                  //对每张碎片的左右边缘相互对比拼接

    vector<Mat> fragments_LR = fragments_LR_Imread("res/fragments_LR/");     //读取左右拼接好的碎片

    vector<vector<Mat>> resection_TB = edge_resection_TB(fragments_LR);                 //保存拼接好的左右碎片的上下边缘

    alignment_and_splicing_TB(fragments_LR, resection_TB);                              //对左右拼接好的碎片的上下边缘相互对比拼接

    Mat result = imread("res/result/0.jpg");
    imshow("Restoration map",result);                                         //展示结果

    waitKey(0);
    return 0;
}

//读取碎片
vector<Mat> fragments_Imread(string files_name){
    vector<string> files;
    glob(std::move(files_name),files);
    vector<Mat> fragments;
    for(auto &file : files){
        fragments.push_back(imread(file));
    }
    return fragments;
}
vector<Mat> fragments_LR_Imread(string files_name){
    vector<string> files;
    glob(std::move(files_name),files);
    vector<Mat> fragments_LR;
    for(auto &file : files){
        fragments_LR.push_back(imread(file));
    }
    return fragments_LR;
}

//保存每张碎片的左右边缘
vector<vector<Mat> > edge_resection_LR(const vector <Mat>& fragments){
    vector<vector<Mat> > resection_LR(fragments.size(), vector<Mat>(2));
    for(int i = 0; i<fragments.size(); i++){
        for(int j = 0; j<2; j++){
            switch (j){
                case 0:     //第 i 张碎片的 左边；  顶点：（0，0）  尺寸：（10 * 第i张碎片的高/行）
                    resection_LR.at(i).at(j) = fragments.at(i)(Rect(0,0,10, fragments.at(i).rows));
                    break;
                case 1:     //第 i 张碎片的 右边；  顶点：（第 i 张碎片的宽/列-10，0）  尺寸：（10 * 第i张碎片的高/行）
                    resection_LR.at(i).at(j) = fragments.at(i)(Rect(fragments.at(i).cols-10,0,10, fragments.at(i).rows));
                default:
                    break;
            }
        }
    }
    return resection_LR;
}

//直方图对比
bool compare_by_hist(const Mat& img1, const Mat& img2){
    Mat tmpImg,orgImg;
    resize(img1, tmpImg, Size(img1.cols, img1.rows));
    resize(img2, orgImg, Size(img2.cols, img2.rows));
    //HSV颜色特征模型(色调H,饱和度S，亮度V)
    cvtColor(tmpImg, tmpImg, COLOR_BGR2HSV);
    cvtColor(orgImg, orgImg, COLOR_BGR2HSV);
    //直方图尺寸设置
    //一个灰度值可以设定一个bins，256个灰度值就可以设定256个bins
    //对应HSV格式,构建二维直方图
    //每个维度的直方图灰度值划分为256块进行统计，也可以使用其他值
    int hBins = 256, sBins = 256;
    int histSize[] = { hBins,sBins };
    //H:0~180, S:0~255,V:0~255
    //H色调取值范围
    float hRanges[] = { 0, 180 };
    //S饱和度取值范围
    float sRanges[] = { 0,255 };
    const float* ranges[] = { hRanges, sRanges };
    int channels[] = { 0,1 };//二维直方图
    MatND hist1, hist2;
    calcHist(&tmpImg, 1, channels, Mat(), hist1,2,histSize, ranges, true, false);
    normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
    calcHist(&orgImg, 1, channels, Mat(), hist2, 2, histSize, ranges, true, false);
    normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());
    double similarityValue = compareHist(hist1, hist2, CV_COMP_CORREL);
//    cout << "相似度：" << similarityValue << endl;
    return similarityValue >= 0.2;
}

//左右拼接
void picture_stitching_LR(const Mat& img1, const Mat& img2){
    Mat result;
    hconcat(img1,img2,result);
    imwrite("res/fragments_LR/"+to_string(n)+".jpg", result);
    n++;
}

//对每张碎片的左右边缘相互对比拼接
void alignment_and_splicing_LR(const vector <Mat>& fragments, const vector<vector<Mat>>& resection_LR){
    for(int i = 0; i<fragments.size()-1; i++){            //第 i 张碎片
        for(int j = 0; j<2; j++){                       //第 i 张碎片的第 j 条边
            for(int k = i; k<fragments.size()-1; k++){    //第 i 张碎片的第 j 条边 与 第 i 张以后碎片的左右边缘对比
                for(int l = 0; l<2; l++){
                    if(compare_by_hist(resection_LR.at(i).at(j),resection_LR.at(k+1).at(l))){
                        if(j>l){            //当j>l时被对比的边缘应该在对比右边
                            picture_stitching_LR(fragments.at(i),fragments.at(k+1));
                        } else if(j<l){     //当j<l时被对比的边缘应该在对比右边
                            picture_stitching_LR(fragments.at(k+1),fragments.at(i));
                        }
                    }
                }
            }
        }
    }
}

//上下拼接
void picture_stitching_TB(const Mat& img1, const Mat& img2){
    Mat result;
    vconcat(img1,img2,result);
    imwrite("res/result/"+to_string(m)+".jpg", result);
    m++;
}

//保存左右拼接好的碎片的上下边缘
vector <vector<Mat>> edge_resection_TB(const vector <Mat>& fragments_LR){
    vector <vector<Mat>> resection_TB(fragments_LR.size(), vector<Mat>(2));
    for(int i = 0; i<fragments_LR.size(); i++){
        for(int j = 0; j<2; j++){
            switch (j){
                case 0:     //第 i 张碎片的 上边缘；  顶点：（0，0）  尺寸：（第i张碎片的宽/列 * 10）
                    resection_TB.at(i).at(j) = fragments_LR.at(i)(Rect(0,0,fragments_LR.at(i).cols, 10));
                    break;
                case 1:     //第 i 张碎片的 下边缘；  顶点：（0，第 i 张碎片的高/行-10）  尺寸：（第i张碎片的宽/列 * 10）
                    resection_TB.at(i).at(j) = fragments_LR.at(i)(Rect(0,fragments_LR.at(i).rows-10, fragments_LR.at(i).cols, 10));
                default:
                    break;
            }
        }
    }
    return resection_TB;
}

//对左右拼接好的碎片进行上下对比拼接
void alignment_and_splicing_TB(const vector <Mat>& fragments_LR, const vector<vector<Mat>>& resection_TB){
    for(int i = 0; i<fragments_LR.size()-1; i++){               //第 i 张碎片
        for(int j = 0; j<2; j++){                               //第 i 张碎片的第 j 条边
            for(int k = i; k<fragments_LR.size()-1; k++){       //第 i 张碎片的第 j 条边 与 第 i 张以后碎片的左右边缘对比
                for(int l = 0; l<2; l++){
                    if(compare_by_hist(resection_TB.at(i).at(j),resection_TB.at(k+1).at(l))){
//                        picture_stitching_TB(fragments_LR.at(i),fragments_LR.at(k+1));
                        if(j>l){                                //当j>l时被对比的边缘应该在对比下边
                            picture_stitching_TB(fragments_LR.at(i),fragments_LR.at(k+1));
                        } else if(j<l){                         //当j<l时被对比的边缘应该在对比上边
                            picture_stitching_TB(fragments_LR.at(k+1),fragments_LR.at(i));
                        }
                    }
                }
            }
        }
    }
}