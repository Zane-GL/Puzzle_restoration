#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

int main(){
    //读取模板图片
    Mat Template_picture = imread("res/Template_picture/016.jpg");
    imshow("dfs",Template_picture);

    //读取拼图碎片
    vector<String> files;
    glob("res/fragment",files);
    vector<Mat> fragments;          //存到一个Mat容器中
    for (const auto & file : files) {
        fragments.push_back(imread(file));
    }

    //保存模板图片的尺寸
    vector<Size> Template_picture_sizes;
    Template_picture_sizes.push_back( Template_picture.size() );
    cout<< "The size of template picture:  "<<Template_picture_sizes.at(0) <<endl;

    //创建一个和模板图片相同大小的图片
    Mat result;
    result.create(Template_picture_sizes.at(0),CV_32FC1);

    //模板匹配
    matchTemplate(Template_picture, fragments.at(0), result, cv::TM_SQDIFF_NORMED);
    normalize(result, result, 0, 1, NORM_MINMAX, -1,Mat());     //匹配结果归一化

    //寻找最佳匹配位置
    double minVal, maxVal;
    Point minLoc, maxLoc;//碎片在对应模板图片的坐标
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    //用矩形画出其位置
    rectangle(Template_picture, minLoc, Point(minLoc.x + fragments.at(0).cols, minLoc.y + fragments.at(0).rows), Scalar(0, 255, 0), 2, 8, 0);

    //将相应位置的碎片移动到模板图片上
    Mat imageROI = Template_picture(Rect(minLoc,fragments.at(0).size()));
    fragments.at(0).copyTo(imageROI);

    imshow("Result",Template_picture);

    waitKey(0);

    return 0;
}


/*int main()
{
    Mat image = imread("res/Template_picture/016.jpg");
    cout<<image.size()<<endl;
    Mat logo = imread("res/fragment/part3.jpg");
    cout<<logo.size()<<endl;
    imshow("原图", image);
    imshow("logo", logo);
    Mat imageROI;
    imageROI = image(Range(0, logo.rows), Range(logo.cols,2*logo.cols));
    imshow("ROI",imageROI);
    logo.copyTo(imageROI);
    imshow("原图+logo", image);
    waitKey(0);
    return 0;

}*/

/*    for(int i = 0; i < fragments.size(); i++){
    matchTemplate(Template_picture, fragments[i], result.at(i),cv::TM_SQDIFF_NORMED);
    normalize(result.at(i), result.at(i), 0, 1, NORM_MINMAX, -1,Mat());
    minMaxLoc(result.at(i), &minVal[i], &maxVal[i], &minLoc[i], &maxLoc[i], Mat());
    matchLoc[i] = minLoc[i];
    rectangle(Template_picture,  matchLoc[i], Point(matchLoc[i].x + fragments[i].cols, matchLoc[i].y + fragments[i].rows), Scalar(0, 255, 0), 2, 8, 0);
    cout<<"匹配度："<<minVal[i]<<endl;
}*/

/*vector<Mat> result;
    double minVal[fragments.size()];
    double maxVal[fragments.size()];
    Point minLoc[fragments.size()];
    Point maxLoc[fragments.size()];
    Point matchLoc[fragments.size()];
    minVal[0] = -1;

    //保存模板图片的尺寸
    vector<int> Template_picture_sizes;
    Template_picture_sizes.push_back(Template_picture.cols);
    Template_picture_sizes.push_back(Template_picture.rows);

    //模板匹配
    matchTemplate(Template_picture, fragments[1], result[1], cv::TM_SQDIFF_NORMED);
    normalize(result.at(1), result.at(1), 0, 1, NORM_MINMAX, -1,Mat());

    minMaxLoc(result.at(1), &minVal[1], &maxVal[1], &minLoc[1], &maxLoc[1], Mat());
    matchLoc[1] = minLoc[1];
    rectangle(Template_picture,  matchLoc[1], Point(matchLoc[1].x + fragments[1].cols, matchLoc[1].y + fragments[1].rows), Scalar(0, 255, 0), 2, 8, 0);



    imshow("img", Template_picture);
    cout<<"匹配度："<<minVal<<endl;*/

/*int main()
{
    //加载源图像和模板图像
    cv::Mat image_source = cv::imread("res/lena.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat image_template = cv::imread("res/template.jpg", cv::IMREAD_GRAYSCALE);

    cv::Mat image_matched;

    //模板匹配
    cv::matchTemplate(image_source, image_template, image_matched, cv::TM_CCOEFF_NORMED);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    //寻找最佳匹配位置
    cv::minMaxLoc(image_matched, &minVal, &maxVal, &minLoc, &maxLoc);

    cv::Mat image_color;
    cv::cvtColor(image_source, image_color, cv::COLOR_GRAY2BGR);
    cv::circle(image_color,
               cv::Point(maxLoc.x + image_template.cols/2, maxLoc.y + image_template.rows/2),
               20,
               cv::Scalar(0, 0, 255),
               2,
               8,
               0);

    cv::imshow("source image", image_source);
    cv::imshow("match result", image_matched);
    cv::imshow("target", image_color);
    cv::waitKey(0);

    return 0;
}*/



