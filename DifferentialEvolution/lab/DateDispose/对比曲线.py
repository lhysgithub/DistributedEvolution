import os

Output = open("Querytimes.txt",'w')
for i in range(16,17):
    if i%10==i//10:
        Output.write('0')
        Output.write('\n')
        continue

    with open("log%d.txt"%i,'r') as f:
        List = f.readlines()
        LastNumber = 0
        Number = 0
        for k in range(0,len(List)):
            Usefully = List[k].split()
            if Usefully[0] == 'Step':
                Output.write(Usefully[3])
                Output.write('\n')
                # break
