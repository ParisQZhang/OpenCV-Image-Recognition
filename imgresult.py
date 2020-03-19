def imgresult = imgmatch(imgfile):
import numpy as np
import cv2 as cv
s1 = 'DeepLearning.JPG';s2 = 'ImageVideo.JPG';
s3 = 'ModenPhoto.JPG'; s4 = 'PatternRecognition.JPG'; 
booklist = [s1,s2,s3,s4];
s11 = 'deep_goodfellow';s22 = 'handbook_bovik';
s33 = 'modern_mikhail'; s44 = 'pattern_bishop'; 
newbook = [s11,s22,s33,s44];
   
for a < len(booklist)
imgRGB = cv.imread(booklist(a));
imgRGB = cv.Resize(imgRGB, [1000,NaN]);
img = single(rgb2gray(imgRGB));
[fcover,dcover] = vl_sift(img);

#imgfile = imread('IMG_0212.JPG');
imgfile = cv.resize(imgfile, [1000,NaN]);
Ia = single(rgb2gray(imgfile));
[fa, da] = vl_sift(Ia) ;

[matches, scores] = vl_ubcmatch(dcover, da) ;

#figure(1) ; clf ;
#imagesc(cat(2, Ia, img)) ;

xa = fa(1,matches(2,:)) ;
xcover = fcover(1,matches(1,:)) + size(Ia,2) ;
ya = fa(2,matches(2,:)) ;
ycover = fcover(2,matches(1,:)) ;

h = line([xa ; xcover], [ya ; ycover]) ;
set(h,'linewidth', 1, 'color', 'b') ;

numMatches = length(scores);
#Get matching x and y values
xycover = [fcover(1:2,matches(1,:));ones(1,numMatches)];
xyimg = [fa(1:2,matches(2,:));ones(1,numMatches)];

#Find most accurate H using RANSAC
num_inliner_best = 0;num_inlier = 0;
BestH = 0;H = [];dist_thr = 2;
max_n_trials = 1000;
for ii=1:max_n_trials
    pt_idx = randperm(numMatches,4);
    coverprand = [];imgprand = [];
    for k = 1:length(pt_idx)
        coverprand = [coverprand;xycover(1:2,pt_idx(k))'];
        imgprand = [imgprand;xyimg(1:2,pt_idx(k))'];
    end
    T = ComputeH(imgprand,coverprand);
   % H = [H,T];
    xycoverH = (xycover'*T');
    xycoverH = xycoverH';
    dx = xycoverH(1,:)./xycoverH(3,:)-xyimg(1,:);
    dy = xycoverH(2,:)./xycoverH(3,:)-xyimg(2,:);
    dist = sqrt(dx.^2+dy.^2);
    num_inlier = sum(dist<dist_thr);
     if num_inlier>num_inliner_best
        BestH = T;
        num_inliner_best = num_inlier;
    end
end

%% booklet cover corners in homogeneous coordinates

covercorners = [1,1,1; size(imgRGB,2),1,1;size(imgRGB,2),size(imgRGB,1),1;1,size(imgRGB,1),1];

%%cover corners on img
cornersH = covercorners*BestH';
cornersH = cornersH./cornersH(:,3);
bookcentre = [mean([min(cornersH(:,1)),max(cornersH(:,1))]),mean([min(cornersH(:,2)),max(cornersH(:,2))])];
%imshow(imgfile),hold on
cornershape = [];
for i = 1:size(cornersH,1)
  cornershape = [cornershape,cornersH(i,1:2)];
end
#for j = 1:size(cornersH,1)
  imgresult = insertShape(imgfile,'FilledPolygon',cornershape,'Color','white','Opacity',0.4);
  imgresult = insertShape(imgresult,'Line',[cornershape,cornersH(1,1:2)],'Color','red','LineWidth',5);
  imgresult = insertText(imgresult,bookcentre,newbook(a),'FontSize',24,'TextColor','white','AnchorPoint','Center','Boxcolor','red');
  imgresult = insertMarker(imgresult,bookcentre,'o','color','blue','Size',5);

end
end
