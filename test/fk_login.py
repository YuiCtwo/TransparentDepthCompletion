from selenium import webdriver
from selenium.webdriver import FirefoxOptions
import time
import subprocess

username_str = "*******"  # 你的校园网登陆用户名
password_str = "********"  # 你的校园网登陆密码

login_url = "http://202.113.18.106/a70.htm?"


# ping_cmd = "ping -c 3 baidu.com | grep '0 received | wc -l'"

def login():
    profile = webdriver.FirefoxProfile()
    # 禁止下载图片
    profile.set_preference("permissions.default.image", 2)
    # 禁用浏览器缓存
    profile.set_preference("network.http.use-cache", False)
    profile.set_preference("browser.cache.memory.enable", False)
    profile.set_preference("browser.cache.disk.enable", False)
    profile.set_preference("browser.sessionhistory.max_total_viewers", 3)
    profile.set_preference("network.dns.disableIPv6", True)
    profile.set_preference("Content.notify.interval", 750000)
    profile.set_preference("content.notify.backoffcount", 3)
    # 有的网站支持 有的不支持 2 35 profile.set_preference("network.http.pipelining", True)
    profile.set_preference("network.http.proxy.pipelining", True)
    profile.set_preference("network.http.pipelining.maxrequests", 32)

    opts = FirefoxOptions()
    opts.add_argument("--headless")
    driver = webdriver.Firefox(options=opts, firefox_profile=profile)

    try:
        driver.get(login_url)  # 你的校园网登陆地址
        driver.implicitly_wait(30)

        username_input = driver.find_elements_by_name("DDDDD")[1]
        password_input = driver.find_elements_by_name("upass")[1]
        login_button = driver.find_elements_by_name("0MKKey")[1]

        if username_input and password_input and login_button:
            print(get_current_time(), 'username and password find successfully')
        username_input.send_keys(username_str)
        password_input.send_keys(password_str)
        login_button.click()
        if check_connection():
            print(get_current_time(), 'login successfully')
    except Exception as e:
        print(get_current_time(), 'login failed, error message: {}'.format(e))
    finally:
        driver.quit()


# 获取当前时间
def get_current_time():
    return time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time()))


# 判断当前是否可以连网
def check_connection():
    ping_caller = subprocess.Popen(["ping", "-c", "3", "baidu.com"], stdout=subprocess.PIPE)
    try:
        result = ping_caller.communicate(timeout=10)[0]
        result = result.decode("utf-8")
        if result.count("ttl") != 3:
            return False
        else:
            return True
    except TimeoutError:
        return False


def loop_connection():
    pass


if __name__ == '__main__':
    login()
    # print(check_connection())
