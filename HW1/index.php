<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>CS362 Homework Demo Page</title>
<link href="../css/normalize.css" media="all" rel="stylesheet" type="text/css" />
<link href="../css/bootstrap.min.css" rel="stylesheet">
<link href='//fonts.googleapis.com/css?family=Roboto:400,300' rel='stylesheet' type='text/css'>
</head>

<body>
<div class='container-fluid'>
<h1>CS362 <?php echo $_GET['std']?>的作業  <a href="http://140.138.152.202/ipDemo/index.php">返回課程網頁</a></h1>

<hr>
<div>
<h2><a href="../hw1.php?std=<?php echo $_GET['std']?>">作業1展示</a><br></h2>
<h4>程式簡介</h4>
&lt;&nbsp;<b>亮度調整程式</b>&nbsp;&gt; <br>
步驟一 : 讀取一張圖片<br>
步驟二 : 讀取使用者想要調整的值 ( inc )<br>
步驟三 : 將圖的圖上半部加上讀取到的數值，圖的下半部減去讀取到的數值<br>
步驟四 : 處理像素值溢位問題<br>
<br>
註1 : 為了解決溢位問題，使用 OpenCV 中的 saturate_cast<br>
註2 : 若是 inc 為負值，則圖片上半部則會加上負值，下半部減負值，因此結果圖為下半部比上半部亮
<p>
<br>
<h4>程式結果</h4>
<pre>
&lt; <b>原始圖片</b> &gt;                                                                    &lt; <b>結果圖片</b> &gt;
</pre>
此結果圖 inc = 50 <br>
<img src= "https://imageshack.com/a/img924/2702/OPP0O8.jpg">
<img src= "https://imageshack.com/a/img922/4365/1X64bq.jpg">
<hr>

<div>
<h2><a href="../hw2.php?std=<?php echo $_GET['std']?>">作業2展示</a><br></h2>
<h4>程式簡介</h4>
&lt;&nbsp;<b>圖片旋轉程式</b>&nbsp;&gt; <br>
步驟一 : 讀取一張圖片<br>
步驟二 : 讀取使用者想要旋轉的值 ( rotate )<br>
步驟三 : 找出原圖的四角座標 ( 以中心為圓心 )<br>
步驟四 : 計算旋轉後的四角座標<br>
步驟五 : 找到最長的當作新的行、列<br>
步驟六 : 計算差值並將每個像素點都移到新座標<br>
步驟七 : 處理超出範圍的問題以及分別處理彩色及灰階圖片<br>
<br>
註1 : 彩色 ( Vec3b )，灰階 ( uchar )<br>
註2 : 輸入的 rotate 數值，正值為逆時針旋轉
<p>
<br>
<h4>程式結果</h4>
<pre>
&lt; <b>原始圖片</b> &gt;                                                                    &lt; <b>結果圖片</b> &gt;
</pre>
此結果圖 rotate = 60 <br>
<img src= "https://i.imgur.com/ojuRZBs.jpg" width = "46%" >
<img src= "https://i.imgur.com/DoA1duT.jpg" width = "46%" height="10%">
<hr>
</div>

<div>
<h2><a href="../hw3.php?std=<?php echo $_GET['std']?>">作業3展示</a><br></h2>
<h4>程式簡介</h4>
&lt;&nbsp;<b>直方圖均化</b>&nbsp;&gt; <br>
步驟一 : 讀取一張照片，並將其轉成灰階<br>
步驟二 : 利用 OpenCV 中的 calcHist 來計算直方圖<br>
步驟三 : 畫出此灰階圖的直方圖<br>
步驟四 : 開始均化，先將像素做轉換<br>
步驟五 : 再建立查找表<br>
步驟六 : 查找表並輸出均化過的圖<br>
步驟七 : 與步驟二相同，畫出均化後的直方圖<br>
<br>
註1 : 畫直方圖時，用 line 和 rectangle 都可<br>
註2 : 若不自己寫，OpenCV 中可呼叫 equalizeHist() 來完成
<p>
<br>
<h4>程式結果</h4>
<pre>
&lt; <b>原始圖片</b> &gt; 
</pre>
<img src= "https://i.imgur.com/NlI5e16.png" width = "20%" >
<br>

<pre>
&lt; <b>輸入的原始圖轉成灰階</b> &gt;                &lt; <b>均化後的圖</b> &gt;
</pre>
<img src= "https://i.imgur.com/JeNmypH.jpg" width = "20%" >&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
<img src= "https://i.imgur.com/YmFVZBl.jpg" width = "20%" ><br>

<pre>
&lt; <b>原始圖畫出的直方圖</b> &gt;                  &lt; <b>均化的圖畫出的直方圖</b> &gt;
</pre> 
<img src= "https://i.imgur.com/Z0en5yw.jpg" width = "20%" >&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src= "https://i.imgur.com/yNfkml4.jpg" width = "20%" ><br>
<hr>
</div>

<div>
<h2><a href="../hw4.php?std=<?php echo $_GET['std']?>">作業4展示</a><br></h2>
<h4>說明</h4>
尚未完成作業敘述與說明
</div>
<hr>

<div>
<h2><a href="../hw5.php?std=<?php echo $_GET['std']?>">作業5展示</a><br></h2>
<h4>說明</h4>
尚未完成作業敘述與說明
</div>
<hr>

<div>
<h2><a href="../hw6.php?std=<?php echo $_GET['std']?>">作業6展示</a><br></h2>
<h4>說明</h4>
尚未完成作業敘述與說明
</div>
<hr>

<div>
<h2><a href="../hw7.php?std=<?php echo $_GET['std']?>">作業7展示</a><br></h2>
<h4>說明</h4>
尚未完成作業敘述與說明
</div>
<hr>

<div>
<h2><a href="../hw8.php?std=<?php echo $_GET['std']?>">作業8展示</a><br></h2>
<h4>說明</h4>
尚未完成作業敘述與說明
</div>
<hr>
	
</div>
</body>
</html>