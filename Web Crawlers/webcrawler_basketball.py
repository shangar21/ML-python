from bs4 import  BeautifulSoup as soup 
from urllib.request import urlopen as ureq
import csv 

my_url = 'http://www.espn.com/college-sports/basketball/recruiting/playerrankings/_/class/2021'
client = ureq(my_url) #opens connection, grabs page
page_html = client.read() 
client.close()
#html parsing
page_soup = soup(page_html, "html.parser")
#grabs all item info via html
containers = page_soup.findAll("tr")

#array format: [[player, pos, [hometown, high school], ht, wt], [player, pos, hometown, ht, wt, stars, grade]]
p_info = []

for c in containers:
	if("oddrow" in c["class"]) or ("evenrow" in c["class"]):
		s = c.findAll("td")
		name = s[1].div.a.strong.text
		pos = s[2].b.text
		hometown = s[3].text
		ht = s[4].text
		wt = s[5].text
		p_info.append([name, pos, hometown, ht, wt])

#string formatting 
for i in p_info:
	index = i[2].find(",") + 4
	i[2] = [i[2][:index], i[2][index:]]
	print([i[2][:index], i[2][index:]])
	 


with open("college_players.csv", 'w', encoding = 'utf-8') as toWrite:
	writer = csv.writer(toWrite)
	writer.writerows(p_info)

print("got the classified info fam")
				

