# IMDB爬虫任务结题总结报告
# 1 任务要求
调用python的selenium库采集[IMDB](https://imdb.com/ "IMDB")网站中的指定影评与对应影评者的相关信息。
- 指定影评id文件：[下载](https://file.zhihuishu.com/zhs/ablecommons/demo/202009/38ed4929a18848e6b804e59b3e7b05f2.txt "下载")
- 参考：[关于 IMDB 要采集的数据说明](https://blog.qsclub.cn/index.php/archives/23/ "关于 IMDB 要采集的数据说明")

# 2 关键问题及解决方案
## 2.1 在影评页面，每次只能获取到25条影评，如何快速获得所有异步加载的影评数据？

![](https://image.zhihuishu.com/zhs/ablecommons/demo/202009/cf7007c1d8de43cba17d7b03d97fb950.png)
### 思路一（失败）：分析js代码以获得该按钮绑定的加载函数，通过js注入反复调用该函数
实践：该网站的js代码已混淆加密，不易找到加载函数。

![](https://image.zhihuishu.com/zhs/ablecommons/demo/202009/561a3ed4505e4c42a1d47e89941c0b0f.png)

### 思路二（成功）：分析数据包请求url是否可以构造
实践：该请求可以构造，且数据包中不含有除影评以外的多余信息，可以大大提高爬取效率。请求结构为：
```python
https://www.imdb.com/title/{movie_id}/reviews/_ajax?paginationKey={key}
```
eg：https://www.imdb.com/title/tt7395114/reviews/_ajax?paginationKey=g4wp7cbkry2dc3qb7guhzmjurtt42bjhzfmxvlnomwklyczuf43o6ss5oeyvvprldj4k5u72hkbmb4pnuykkcxnwbgmbvka

1. 当key为空时，请求【数据包1】得到该电影下的第1-25条（第1组）影评，且于`.load-more-data`元素包含了【数据包2】的key信息
2. 请求【数据包2】，获得第26-50条（第2组）影评，且于尾部包含了【数据包3】的key信息
3. 以此类推，【数据包n-1】中包含了【数据包n】所需要的key信息（与单链表形式相似）
4. 若【数据包n】为最后一个数据包，则`.load-more-data`元素为，该电影爬取完毕

![](https://image.zhihuishu.com/zhs/ablecommons/demo/202009/1c2ea3b1e2bc4556a16cd10b1d23e254.png)

## 2.2 csv文件存储数据时单元格溢出
由于影评内容（review_text)中多含有逗号、单引号及双引号，而csv文件默认分割符为逗号，字符串又由双引号包裹，故此引发csv单元格溢出问题（与数据库注入原理相似）

![](https://image.zhihuishu.com/zhs/ablecommons/demo/202009/4bec52e3e19c4c309e4260eda7120add.png)
### 思路一（失败）：将review_text中的引号及逗号全部转义
实践：该方法解决了大部分溢出问题，但仍有少量溢出，猜测是其他特殊字符造成的
### 思路二（成功）：更改csv分割符为转义字符 【SunNan】
实践：在定义csv.writer时设置参数delimiter值为'\t'，自然影评数据中不会含有转义字符，该思路成功
```python
# csvFile初始化
cf = open('data_pro.csv', 'w+', encoding='utf-8', newline='')
csv_writer = csv.writer(cf, delimiter='\t')
```

## 2.3 Unicode编码问题（未解决）

- 对一些“带声调的字母”“表情”等特殊unicode字符，在utf-8编码的csv或数据库中会首先解析成汉字
- 该问题与python的数据类型自动转换有关
- 虽然通过excel手动操作使其在csv中显示正常，但数据库中的显示问题还未能解决

![](https://image.zhihuishu.com/zhs/ablecommons/demo/202009/845a57ebc11b488ea8d34efee56b1a92.png)
![](https://image.zhihuishu.com/zhs/ablecommons/demo/202009/fe1b773493144efd903d47d062de8b72.png)

## 2.4 在用户信息页面，requests直接请求会404，selenium访问频繁会503
- requests直接请求前期有1/3概率会404，后期全部404/503 **【JiaKui】**
- selenium如果不进行休息，连续访问会导致503

![](https://image.zhihuishu.com/zhs/ablecommons/demo/202009/7a562b31f18a42f38409e26ecafbc0bf.png)![](https://image.zhihuishu.com/zhs/ablecommons/demo/202009/9a03c847e564435580fd7ada4be4c3ce.png)

### 解决办法：间歇爬取与隐式等待
```python
# 休息0-5s之间的随机数，模拟用户访问
time.sleep(random.random() * 5)
# 访问用户主页user_page
driver.get(user_page)
# 设置隐式等待最长超时时间为20s
WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'sidebar')))
```
## 2.5 其他
其余问题不具代表价值，不在话下。

# 3 如何加快爬取速度
### 3.1 分布式
![](https://image.zhihuishu.com/zhs/ablecommons/demo/202009/e1992319b6d34e34a96c8f32649489fc.png)

**3.1.1 意义：通过增加爬取主机的数量提升爬取速度**

**3.1.2 思路：**
①.从mysql数据库获得一个尚未爬取的用户id(IMDB_id)，并对该用户id进行标记（表示正在爬取，避免其他主机重复爬取）
②.根据用户id爬取其相关信息
③.更新mysql中该用户id的相关信息
④ 获取新的用户id并开始下一轮爬取

**3.1.3 代码：**
```python
import pymysql
import re
import time, random
from lxml import etree
from selenium import webdriver
from selenium.webdriver.common.by import By
# WebDriverWait类是WebDriver提供的等待方法。在设置时间内，默认每隔一段时间检测一次当前页面元素是否存在，如果超过设置时间检测不到就抛出异常。
from selenium.webdriver.support.ui import WebDriverWait
# 将expected_condtions重命名为EC
from selenium.webdriver.support import expected_conditions as EC

def get_user_info(user_id):
    '''返回rating与review和data构成的元组'''
    # 数据初始化
    ratings = 0
    reviews = 0
    # 访问用户主页
    user_page = f'https://www.imdb.com/user/{user_id}/'
    try:
        time.sleep(random.random() * 5)
        driver.get(user_page)
        # 设置隐式等待最长超时时间为20s
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'sidebar')))
    except:
        print('用户可能不存在，跳过')
        return 0, 0, ''
    html = etree.HTML(driver.page_source)
    print('用户主页：', end='')
    try:
        date = html.xpath('//*[@class="timestamp"]/text()')[0]
        date = date[date.find('since') + 6:]
        print('date获取成功')
    except:
        date = 'unknown'
        print('date获取失败')
    for text in html.xpath('//*[@class="see-more"]/a/text()'):
        if 'ratings' in text:
            ratings = int(''.join(re.findall('\d+', text)))
            print('用户主页：ratings获取成功')
            continue
        if 'reviews' in text:
            reviews = int(''.join(re.findall('\d+', text)))
            print('用户主页：reviews获取成功')
            continue
    if ratings == 0:
        inactive = html.xpath('//*[@class="subNavItem inactive"]/text()')
        while True:
            # 如果存在不能点击的链接 且 第一个不能点击的链接为'Ratings'，则ratings=0。否则访问评分页获取
            if (inactive != []) and (inactive[0] == 'Ratings'):
                print('用户主页：ratings为0')
                break
            else:
                try:
                    time.sleep(random.random() * 5)
                    driver.get(f'https://www.imdb.com/user/{user_id}/ratings')
                    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'main')))
                    rating = driver.find_element_by_xpath('//*[@id="lister-header-current-size"]').text.replace(',', '')
                    ratings = int(''.join(re.findall('\d+', rating)))
                    print('评分页：ratings获取成功')
                    break
                except: pass
    if reviews == 0:
        print('评论页：', end='')
        while True:
            try:
                time.sleep(random.random() * 5)
                driver.get(f'https://www.imdb.com/user/{user_id}/comments-index')
                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'main')))
                review = driver.find_element_by_xpath('//*[@id="main"]/section/div/div[1]/div/span').text.replace(',', '')
                reviews = int(''.join(re.findall('\d+', review)))
                print('reviews获取成功')
                break
            except: pass
    return ratings, reviews, date
def update_data(IMDB_ID, ratings, reviews, IMDB_member, record):
    '''
    数据入库函数，谨慎修改！！！
    :param IMDB_ID:用户id
    :param ratings:评分数量
    :param reviews:评论数量
    :param IMDB_member:注册年月
    :return:空
    '''
    add_sql = f"""UPDATE `user` SET ratings={ratings}, reviews={reviews}, IMDB_member='{IMDB_member}', record='{record}' WHERE IMDB_ID='{IMDB_ID}'"""
    print(add_sql)
    cur.execute(add_sql)
    conn.commit()
def connect():
    '''获取数据库连接（谨慎修改！！！)'''
    while True:
        try:
            conn = pymysql.connect(host='主机名', user='用户名', passwd='密码', db='数据库名', port=3306, charset='utf8')
            cur = conn.cursor()
            break
        except: pass
    print('数据库链接成功！')
    return conn,cur


# webdriver初始化
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_experimental_option('excludeSwitches', ['enable-logging'])
options.add_argument('--disable-gpu')
# options.add_argument('--proxy-server=http://18.138.254.5:80')
driver = webdriver.Chrome(options=options)
# driver.get('http://httpbin.org/ip')

# pymysql初始化
conn, cur = connect()
# 设置初始偏移量
while True:
    name = input('您的昵称（长度<20）：')
    if len(name) <= 20: break
i = int(input('设置初始偏移量：'))
while True:
    print()
    #查询数据（谨慎修改！！！)
    while True:
        try:
            cur.execute(f"SELECT * FROM `user` LIMIT 1 OFFSET {i-1}")
            break
        except: conn, cur = connect()
    #获取一行
    row_1 = cur.fetchone()
    # print(row_1)
    IMDB_ID, ratings, reviews, IMDB_member = row_1[1], row_1[2], row_1[3], row_1[4]
    record = row_1[5]
    if reviews == 0:
        print(i, IMDB_ID, f'爬取人：{name}')
        # 表示正在爬取，如果数据库链接中断则尝试重连
        while True:
            try:
                update_data(IMDB_ID, -1, -1, '', name)
                break
            except: conn, cur = connect()
        ratings, reviews, IMDB_member = get_user_info(IMDB_ID)
        # 爬取结束则更新数据，如果数据库链接中断则尝试重连
        while True:
            try:
                update_data(IMDB_ID, ratings, reviews, IMDB_member, name)
                break
            except: conn, cur = connect()
    else:
        print(i, IMDB_ID, f'爬取人：{record}')
    # #获取多个(3)行
    # row_2 = cur.fetchmany(3)
    # print(row_2)
    # #获取所有，元组形式返回
    # row_3 = cur.fetchall()
    # print(row_3)
    i += 1
    # 若遍历结束，则跳出循环，程序结束
    if row_1 == None: break

# pymysql关闭
cur.close()
conn.close()

# 结束
input('获取完成，按回车键结束...')
```

### 3.2 IP池+多进（线）程 【SunNan】
**多线程要点:**
若不考虑ip访问限制，线程最多可以开到计算机cpu满载为止，大概5000到2万
尝试访问了一个获取pdf网站，线程产生数设置为10万，但达到平衡时线程存在数为一万。所以以后爬取速度的上限限制就是ip池的数量了
若考虑ip访问限制，两种方法：
1.	找到ip复用的最小周期，利用ip池数计算出产生线程的最大速度
2.	高匿ip访问网站后前10秒左右质量极高，可以利用此先存储大量有效ip然后不等待，开大量线程直到ip用尽503，但访问成功率下降至一半。（目前尝试下限为5000个线程并发）
防止线程不断增加：加线程锁`th. Semaphore(500)`

```python
# 核心CODE
sem = th.Semaphore(500)
for i in range(n):
	sem.acquire()
	Program1=th.Thread(target=,args=)
	time.sleep(t)
	Program1.start()
```

### 3.3 自建php代理
![](https://image.zhihuishu.com/zhs/ablecommons/demo/202009/5ace171a09b94a3e8971ce1d5c319831.gif)
**3.3.1 PHProxy说明**
> PHProxy是用PHP编写的Web HTTP代理。它旨在通过与流行的CGIProxy非常相似的Web界面绕过代理限制，PHProxy唯一需要的是安装了PHP的Web服务器。

Github地址:https://github.com/PHProxy/phproxy
示例：http://p2.qsclub.cn/bak/
由于当前市面上充斥大量劣质idc推出的“免费主机”服务，我们可以利用这些主机搭建PHProxy服务以使用其ip访问目标页面。
**3.3.2 代码**
```python
import base64
import pymysql
import re
import time, random
from lxml import etree
from selenium import webdriver
from selenium.webdriver.common.by import By
# WebDriverWait类是WebDriver提供的等待方法。在设置时间内，默认每隔一段时间检测一次当前页面元素是否存在，如果超过设置时间检测不到就抛出异常。
from selenium.webdriver.support.ui import WebDriverWait
# 将expected_condtions重命名为EC
from selenium.webdriver.support import expected_conditions as EC

def b64(url):
    global proxy
    url = f'http://{proxy}/index.php?_proxurl={base64.b64encode(url.encode()).decode()}'
    # print(url)
    return url
def get_user_info(user_id):
    '''返回rating与review和data构成的元组'''
    # 数据初始化
    ratings = 0
    reviews = 0
    # 访问用户主页
    user_page = b64(f'https://www.imdb.com/user/{user_id}/')
    try:
        time.sleep(random.random() * 5)
        driver.get(user_page)
        # 设置隐式等待最长超时时间为20s
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'sidebar')))
    except:
        print('用户可能不存在，跳过')
        return 0, 0, ''
    html = etree.HTML(driver.page_source)
    print('用户主页：', end='')
    try:
        date = html.xpath('//*[@class="timestamp"]/text()')[0]
        date = date[date.find('since') + 6:]
        print('date获取成功')
    except:
        date = 'unknown'
        print('date获取失败')
    for text in html.xpath('//*[@class="see-more"]/a/text()'):
        if 'ratings' in text:
            ratings = int(''.join(re.findall('\d+', text)))
            print('用户主页：ratings获取成功')
            continue
        if 'reviews' in text:
            reviews = int(''.join(re.findall('\d+', text)))
            print('用户主页：reviews获取成功')
            continue
    if ratings == 0:
        inactive = html.xpath('//*[@class="subNavItem inactive"]/text()')
        while True:
            # 如果存在不能点击的链接 且 第一个不能点击的链接为'Ratings'，则ratings=0。否则访问评分页获取
            if (inactive != []) and (inactive[0] == 'Ratings'):
                print('用户主页：ratings为0')
                break
            else:
                try:
                    time.sleep(random.random() * 5)
                    driver.get(b64(f'https://www.imdb.com/user/{user_id}/ratings'))
                    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'main')))
                    rating = driver.find_element_by_xpath('//*[@id="lister-header-current-size"]').text.replace(',', '')
                    ratings = int(''.join(re.findall('\d+', rating)))
                    print('评分页：ratings获取成功')
                    break
                except: pass
    if reviews == 0:
        print('评论页：', end='')
        while True:
            try:
                time.sleep(random.random() * 5)
                driver.get(b64(f'https://www.imdb.com/user/{user_id}/comments-index'))
                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'main')))
                review = driver.find_element_by_xpath('//*[@id="main"]/section/div/div[1]/div/span').text.replace(',', '')
                reviews = int(''.join(re.findall('\d+', review)))
                print('reviews获取成功')
                break
            except: pass
    return ratings, reviews, date
def update_data(IMDB_ID, ratings, reviews, IMDB_member, record):
    '''
    数据入库函数，谨慎修改！！！
    :param IMDB_ID:用户id
    :param ratings:评分数量
    :param reviews:评论数量
    :param IMDB_member:注册年月
    :return:空
    '''
    add_sql = f"""UPDATE `user` SET ratings={ratings}, reviews={reviews}, IMDB_member='{IMDB_member}', record='{record}' WHERE IMDB_ID='{IMDB_ID}'"""
    print(add_sql)
    cur.execute(add_sql)
    conn.commit()
def connect():
    '''获取数据库连接（谨慎修改！！！)'''
    while True:
        try:
            conn = pymysql.connect(host='主机名', user='用户名', passwd='密码', db='数据库名', port=3306, charset='utf8')
            cur = conn.cursor()
            break
        except: pass
    print('数据库链接成功！')
    return conn,cur


# webdriver初始化
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_experimental_option('excludeSwitches', ['enable-logging'])
options.add_argument('--disable-gpu')
# options.add_argument('--proxy-server=http://18.138.254.5:80')
driver = webdriver.Chrome(options=options)
# driver.get('http://httpbin.org/ip')

# pymysql初始化
conn, cur = connect()
# 设置初始偏移量
proxy = input('代理网址：')
name = input('您的昵称；')
i = int(input('设置初始偏移量：'))
while True:
    print()
    #查询数据（谨慎修改！！！)
    while True:
        try:
            cur.execute(f"SELECT * FROM `user` LIMIT 1 OFFSET {i-1}")
            break
        except: conn, cur = connect()
    #获取一行
    row_1 = cur.fetchone()
    # print(row_1)
    IMDB_ID, ratings, reviews, IMDB_member = row_1[1], row_1[2], row_1[3], row_1[4]
    record = row_1[5]
    if reviews == 0:
        print(i, IMDB_ID, f'爬取人：{name}')
        # 表示正在爬取，如果数据库链接中断则尝试重连
        while True:
            try:
                update_data(IMDB_ID, -1, -1, '', name)
                break
            except: conn, cur = connect()
        ratings, reviews, IMDB_member = get_user_info(IMDB_ID)
        # 爬取结束则更新数据，如果数据库链接中断则尝试重连
        while True:
            try:
                update_data(IMDB_ID, ratings, reviews, IMDB_member, name)
                break
            except: conn, cur = connect()
    else:
        print(i, IMDB_ID, f'爬取人：{record}')
    # #获取多个(3)行
    # row_2 = cur.fetchmany(3)
    # print(row_2)
    # #获取所有，元组形式返回
    # row_3 = cur.fetchall()
    # print(row_3)
    i += 1
    # 若遍历结束，则跳出循环，程序结束
    if row_1 == None: break

# pymysql关闭
cur.close()
conn.close()

# 结束
input('获取完成，按回车键结束...')
```
**3.3.3 以下是本次任务过程中搭建的20个代理站点（可用于访问谷歌或加速）**

| 域名                                           | ip 地址               |
| ------------------------------------------------ | ----------------------- |
| http://p2.qsclub.cn/bak                          | 23.224.53.118           |
| http://p3.qsclub.cn/bak                          | 47.244.105.35           |
| http://p4.qsclub.cn/bak                          | 154.222.20.175          |
| http://p5.qsclub.cn/bak                          | 193.243.164.24          |
| http://baidu.shujuidc.cn/bak                     | 91.121.81.113           |
| http://p7.qsclub.cn/bak                          | 27.50.54.215            |
| http://p8.qsclub.cn/bak                          | 47.254.129.86           |
| http://p9.qsclub.cn/bak                          | 47.52.117.26            |
| http://asm143pt91.meiguo.freehost.8800fk.top/bak | 185.245.0.124           |
| http://p11.qsclub.cn/bak                         | 202.5.28.206            |
| http://proxy.wecai88.com/bak                     | 204.152.210.26          |
| http://p13.qsclub.cn/bak                         | 172.247.123.235         |
| http://p14.qsclub.cn/bak                         | 103.229.183.199         |
| http://p15.qsclub.cn/bak                         | 106.54.14.138（国内） |
| http://p16.qsclub.cn/bak                         | 47.244.190.14           |
| http://p17.qsclub.cn/bak                         | 212.95.145.129          |
| http://p18.qsclub.cn/bak                         | 47.56.233.119           |
| http://p19.qsclub.cn/bak                         | 193.243.164.214 (同 p5) |
| http://p20.qsclub.cn/bak                         | 149.202.100.202         |
| http://p21.qsclub.cn/bak                         | 45.195.153.138          |
| http://p22.qsclub.cn/bak                         | 27.50.54.158            |

# 4 技术总结 【HaoLin】
### 4.1 爬虫基础
**4.1.1 Requests**
①.使用fake_useragent 库可以方便快速随机产生多种浏览器头
```python
from fake_useragent import UserAgent  
headers={'User-Agent': UserAgent().random} 
```
②.设置proxy与timeout
```python
ip = '192.168.1.101:8000'
proxies = {
	'http':f'http://{ip}',
	'https':f'http://{ip}'
}
res = requests.get(url,timeout=timelimit,proxies=proxies,headers=headers)
```
**4.1.2 xpath**
①.XPath 使用路径表达式在 XML 文档中选取节点。节点是通过沿着路径或者 step 来选取的。以下是常用路径表达式：

| 表达式 | 描述 |
| ------------ | ------------ |
| nodename | 选取此节点的所有子节点 |
| / | 从根节点选取 |
| // | 从匹配选择的当前节点选择文档中的节点，而不考虑它们的位置。 |
| . | 选取当前节点。 |
| .. | 选取当前节点的父节点。 |
| @ | 选取属性。 |

②.谓语（Predicates）
谓语用来查找某个特定的节点或者包含某个指定的值的节点。
谓语被嵌在方括号中。
在下面的表格中，我们列出了带有谓语的一些路径表达式，以及表达式的结果：

| 路径表达式                     | 结果                                                                                    |
| ----------------------------------- | ----------------------------------------------------------------------------------------- |
| /bookstore/book[1]                  | 选取属于 bookstore 子元素的第一个 book 元素。                               |
| /bookstore/book[last()]             | 选取属于 bookstore 子元素的最后一个 book 元素。                            |
| /bookstore/book[last()-1]           | 选取属于 bookstore 子元素的倒数第二个 book 元素。                         |
| /bookstore/book[position()<3]       | 选取最前面的两个属于 bookstore 元素的子元素的 book 元素。             |
| //title[@lang]                      | 选取所有拥有名为 lang 的属性的 title 元素。                                |
| //title[@lang='eng']                | 选取所有 title 元素，且这些元素拥有值为 eng 的 lang 属性。            |
| /bookstore/book[price>35.00]        | 选取 bookstore 元素的所有 book 元素，且其中的 price 元素的值须大于 35.00。 |
| /bookstore/book[price>35.00]//title | 选取 bookstore 元素中的 book 元素的所有 title 元素，且其中的 price 元素的值须大于 35.00。 |

**4.1.3 ip代理 【JiaKui】**
Ⅰ.得到简易ip池的方法：
①.获取大量ip
②.检验ip是否有用，并判断是否高匿
③.清洗非可用高匿ip，形成ip池

Ⅱ.维护简易ip池的方法：
①.定期对ip池内ip进行检测，若ip失效，则抛弃
②.定期注入新的可用高匿ip

Ⅲ.如何判断是否为高匿ip：
识别的办法就是抓数据包里的字段：REMOTE_ADDR，HTTP_VIA、HTTP_X_FORWARDED_FOR。
①.透明代理
REMOTE_ADDR = Proxy IP
HTTP_VIA = Proxy IP
HTTP_X_FORWARDED_FOR = Your IP
②.普通匿名代理
REMOTE_ADDR = proxy IP
HTTP_VIA = proxy IP
HTTP_X_FORWARDED_FOR = proxy IP
③.高匿代理
REMOTE_ADDR = Proxy IP
HTTP_VIA = not determined
HTTP_X_FORWARDED_FOR = not determined
透明代理会向目标服务器透露自己的真实IP，普匿代理会向目标服务器透露用了代理，髙匿代理什么都不透露给目标服务器。

Ⅳ.代理ip有效性检测API：http://httpbin.org/ip

Ⅴ.代理ip服务评测：
- 付费：
经验尚不丰富，本次使用了站大爷的短效优质ip，体验不错。（评测：https://zhuanlan.zhihu.com/p/33576641 ）
- 免费：
   目前实现的采集免费代理网站有(排名不分先后, 下面仅是对其发布的免费代理情况, 付费代理测评可以参考[这里](https://zhuanlan.zhihu.com/p/33576641)):
![](https://image.zhihuishu.com/zhs/ablecommons/demo/202009/810e87829e0e49e99b955ef0d06f1c0c.png)

### 4.2 多线程与多进程
**4.2.1 多线程**
```python
import threading
import time

def run(n):
    print("task", n)
    time.sleep(1)
    print('2s')
    time.sleep(1)
    print('1s')
    time.sleep(1)
    print('0s')
    time.sleep(1)

if __name__ == '__main__':
    t1 = threading.Thread(target=run, args=("t1",))
    t2 = threading.Thread(target=run, args=("t2",))
    t1.start()
    t2.start()
```
**4.2.2 多进程**
```python
import time
import random
from multiprocessing import Process

def run(name):
    print('%s runing' %name)
    time.sleep(random.randrange(1,5))
    print('%s running end' %name)

p1=Process(target=run,args=('anne',)) #元组格式传入
p2=Process(target=run,args=('alice',))
p3=Process(target=run,args=('biantai',))
p4=Process(target=run,args=('haha',))

p1.start()
p2.start()
p3.start()
p4.start()
```
# 5 感悟与收获

### SunNan
1. 多人分工合作，同时对难题尝试各自擅长的方法是很好的解决问题方法。既能加大成功率又能从各种方向发现问题所在。
2. 做项目尽量不用Spyder，用jupyter更好
3. 该花钱时就花钱，没学费就没质量☹
4. 学会了mysql基本操作原理，多线程与多进程，加强了一些前端知识和代理服务器认识

### JiaXu
1. 接触到了较为严格的反爬机制，改变了“requests爬所有”的旧观点；
2. 对mysql的基本操作更为熟悉，学习了如何用pymysql库连接mysql并执行语句；
3. 了解了selenium的webdriver方法，以及显示等待与隐式等待的优缺点；
4. 学习了xpath基本语法；
5. 实践了分布式爬取的思想；
6. 对代理ip、ip池有了更深入的了解；
7. 入门多进程与多线程，了解了两者的差别与联系；
8. 体验团队协作，感觉良好；