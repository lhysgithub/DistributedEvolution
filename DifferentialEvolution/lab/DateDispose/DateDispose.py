import os

Output = open("UseingStep.txt",'w')
for i in range(0,100):
    if i%10==i//10:
        Output.write('0')
        Output.write('\n')
        continue

    with open("log%d.txt"%i,'r') as f:
        List = f.readlines()
        for k in range(1,len(List)):
            Usefully = List[-k].split()
            if Usefully[0] == 'Step':
                Output.write(Usefully[1][:-1])
                Output.write('\n')
                break
        # for k in range(len(Usefully)):
        #     if Usefully[k] == "QueryTimes:":
        #         Output.write(Usefully[k+1])
        #         Output.write('\n')
    # with open("log%d.txt"%i,'r') as f:
    #     List = f.readlines()
    #     Usefully = List[-1]
    #     Usefully = Usefully.split()
    #     for k in range(len(Usefully)):
    #         if Usefully[k] == "QueryTimes:":
    #             Output.write(Usefully[k+1])
    #             Output.write('\n')

