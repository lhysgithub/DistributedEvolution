import os

Output = open("NeedEpoch.txt",'w')
for i in range(0,100):
    if i%10==i//10:
        Output.write('0')
        Output.write('\n')
        continue

    with open("log%d.txt"%i,'r') as f:
        List = f.readlines()
        number = 0
        for Usefully in List:
            Usefully = Usefully.split()
            if Usefully[0]=='count:' and Usefully[1]=='0':
                continue
            elif Usefully[0]=='count:' and Usefully[1] != '0'and Usefully[1] != '1':
                number += 1
            elif Usefully[0]=='Step':
                number+=1
                Output.write(str(number))
                Output.write('\n')
                break
            # for k in range(len(Usefully)):
            #     if Usefully[k] == "QueryTimes:":
            #         Output.write(Usefully[k+1])
            #         Output.write('\n')
