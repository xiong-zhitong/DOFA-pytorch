import pandas as pd

def get_sinfo():
    return pd.read_csv('sinfo.csv', delimiter=' ')

def get_squeue():
    # read in squeue.csv, turn the first 8 spaces into |
    delim = '@'
    with open('squeue.csv', 'r') as file:
        lines = file.readlines()
    for i in range(len(lines)):
        fields = 0
        out = []
        for c in lines[i]:
            if c == ' ' and fields < 8:
                out.append(delim)
                fields += 1
            else:
                out.append(c)
        lines[i] = ''.join(out)
    with open('squeue.csv', 'w') as file:
        file.writelines(lines)

    return pd.read_csv('squeue.csv', delimiter=delim).drop(columns=['Unnamed: 0'])

def calc_usage(sinfo=None, sque=None, partitions_of_interest = ['savio3_gpu','savio4_gpu']):

    sinfo = sinfo or get_sinfo()
    sque = sque or get_squeue()

    # extract all nodes per partition
    parts = {}
    for i in range(sinfo.shape[0]):
        part_name = sinfo['PARTITION'][i]
        if part_name in partitions_of_interest:
            nodes = parts.get(part_name, set())
            lnodes = sinfo['NODELIST'][i].split(',')
            for n in lnodes:
                nodes.add(n)
            parts[part_name] = nodes
    part2nodes = parts

    # extract unavailable nodes
    unavailable_states = ['down', 'drain','inval','maint','reboot','power','reserv','unknown']
    df = sinfo[  (sinfo['PARTITION'].isin(partitions_of_interest)) ]

    idx = []
    for i in range(df.shape[0]):
        if any([state in df['STATE'].iloc[i] for state in unavailable_states]):
            idx.append(i)
    df = df.iloc[idx]

    part2unavailable = {}
    for i in range(df.shape[0]):
        part_name = df['PARTITION'].iloc[i]
        n = df['NODES'].iloc[i]
        part2unavailable[part_name] = part2unavailable.get(part_name, 0) + n

    # extract all nodes allocated by running jobs
    running = sque[ (sque['ST'] == 'R') ]
    running_nodes = set()
    for r in running['NODELIST(REASON)']:
        lnodes = r.split(',')
        for n in lnodes:
            running_nodes.add(n)

    # build table
    out = []
    for part in part2nodes:
        nodes = part2nodes[part]
        total = len(nodes)
        used = len(nodes.intersection(running_nodes))
        nunav = part2unavailable.get(part, 0)
        idle = total - used - nunav
        out.append([part, total, used, idle, nunav])
    return pd.DataFrame(out, columns=['Partition', 'Total', 'Used', 'Idle', 'Unavailable'])

if __name__ == '__main__':
    df = calc_usage()
    print(df)