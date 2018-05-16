import os

Output = open("QueryTimes.txt",'w')
for i in range(0,10):
    if i%10==i//10:
        Output.write('0')
        Output.write('\n')
        continue

    with open("log%d.txt"%i,'r') as f:
        List = f.readlines()
        Usefully = List[-1]
        Usefully = Usefully.split()
        for k in range(len(Usefully)):
            if Usefully[k] == "QueryTimes:":
                Output.write(Usefully[k+1])
                Output.write('\n')

