
def stock_dl(symbol,start,end):

	import yahoo_finance
	import csv

	for j in symbol:
		stock = yahoo_finance.Share(j)
		history = stock.get_historical(start,end)

		with open("stock/{}.csv".format(j), "w") as toWrite:
		    writer = csv.writer(toWrite,delimiter=",")
		    writer.writerow(['volume','symbol','adj_close','high','low','date','close','open'])
		    for i in history:
		    	temp = []
		    	for a in i.keys():
		    		temp.append(i[a])
		        writer.writerow(temp)