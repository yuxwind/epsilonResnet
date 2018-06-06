import tensorflow as tf
import re, math
import argparse
import os, sys

re_NAME_CIFAR = "res(\d).(\d+)"
re_NAME_IMAGENET = "group(\d)/block(\d+)"
fmt_NAME_CIFAR = "res%d.%d"
fmt_NAME_IMAGENET = "group%d/block%d"
fmt_saved_model = '%s/compressed_model_%d'
group_CIFAR = [1,2,3]
group_IMAGENET = [0,1,2,3]

kept_variable = ['conv0', 'bnlast', 'input_queue_size','linear', 'global_step']
cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]}
fmt_discarded = '/is_discarded: '
fmt_global_step = '(global_step '
discarded_treshold = 0.5


# read log.log and extract discarded blocks
def get_discarded_block(logfile, step):
    discarded_block = []
    step_found = False
    is_cifar_model = True
    N = 0
    idx = 0
    discarded_cnt = 0
    val_error = []
    for l in open(logfile, 'r'):
        if idx == 0:
            idx = 1
            if 'cifar' in l:
                is_cifar_model = True
                re_NAME = re_NAME_CIFAR
                rst = re.search('-n ?(\d+)', l, re.IGNORECASE)
                if rst:
                    N = int(rst.group(1))
                else:
                    print("N could not be figured out in log.log")
                    sys.exit()
            elif 'imagenet' in l:
                is_cifar_model = False
                re_NAME = re_NAME_IMAGENET
                rst = re.search('-d ?(\d+)', l, re.IGNORECASE)
                if rst:
                    N = int(rst.group(1))
                else:
                    print("N could not be figured out in log.log")
                    sys.exit()
            else:
                print('the dataset is unknown')
                sys.exit()
        elif '%s%d'%(fmt_global_step,step) in l:
            step_found = True
        elif step_found:
            if fmt_global_step in l:
                print(l)
                step_found = False
                break
            elif 'discarded_cnt: ' in l:
                print(l)
                discarded_cnt = float(l.strip().split('discarded_cnt: ')[1])
                discarded_cnt = int(math.floor(discarded_cnt))
            elif ' val_error: ' in l:
                val_error.append(float(l.strip().split(' val_error: ')[1]))
            elif ' val-error-top1: ' in l:
                val_error.append(float(l.strip().split(' val-error-top1: ')[1]))
            elif ' val-error-top5: ' in l:
                val_error.append(float(l.strip().split(' val-error-top5: ')[1]))
            else:
                if '%s'%fmt_discarded in l:
                    rst = re.search('(%s)%s(\d(.\d+(e-\d+)?)?)'%(re_NAME,fmt_discarded), 
                            l, re.IGNORECASE)
                    if rst:
                        if float(rst.group(4)) >= discarded_treshold:
                            print(l)
                            discarded_block.append(rst.group(1))
    #todo: what about if is_discarded in (0,1)
    #assert math.floor(discarded_cnt) == len(discarded_block), \
    #        'discarded_cnt and len(discarded_block) are unequal: {}, {}'.format(\
    #        discarded_cnt, len(discarded_block))
    return N, is_cifar_model, discarded_block, val_error
               
def setup(model_dir, step):
    iter_dir = [x for x in os.walk(model_dir)]
    log_cnt = 0
    for root, subdir, files in os.walk(model_dir):
        if 'log.log' in files:
            model_dir = root
            log_cnt = log_cnt + 1
            print('log.log is found at {}'.format(model_dir))
    if log_cnt == 0 or log_cnt > 1:
        print('No log.log or mulitple log.log exists')
        sys.exit()

    log_path = '{}/log.log'.format(model_dir)    
    chk_path = '{}/checkpoint'.format(model_dir)
    model_path = '{}/model-{}.data-00000-of-00001'.format(model_dir, step)
    if not os.path.exists(model_path):
        print('the model file does not exist: model-{}.data-00000-of-00001'.format(step))
        sys.exit()
    if os.path.exists(chk_path):
        os.rename(chk_path, '{}.bak'.format(chk_path))
    with open(chk_path, 'w') as f:
        f.write('model_checkpoint_path: \"model-%d\"\n'%(step))
        f.write('all_model_checkpoint_paths: \"model-%d\"\n'%(step))
    N, is_cifar_model, discarded_block, val_error = get_discarded_block(log_path, step)

    name_mapping, discard_first_block, structure = remap_variable(discarded_block, is_cifar_model, N)
    gen_cfg(is_cifar_model, model_dir, N, discarded_block, step, discard_first_block, structure, val_error)
    return model_dir, is_cifar_model, name_mapping

def clear(model_dir):
    chk_path_bak = '{}/checkpoint.bak'.format(model_dir)
    chk_path= '{}/checkpoint'.format(model_dir)
    if os.path.exists(chk_path_bak):
        os.rename(chk_path_bak, chk_path)

def get_structure(is_cifar_model, N):
    if is_cifar_model:
        if N not in [18, 33, 83, 125]:
            print "#block {} per group of ResNet is invalid".format(N)
            sys.exit()
        # structure[0] is useless but a placeholder.
        structure = [N]*4

    else:
        try:
            structure = cfg[N]
        except:
            print "Depth {} of the ResNet is invalid".format(N)
            sys.exit()
    return structure

def get_block_name(is_cifar_model, name):
    if is_cifar_model:
        return name.split('/')[0]
    else:
        return '/'.join(name.split('/')[:2])

# To fit the model after discarding certain layers, the remained names needs to be updated
def remap_variable(discarded_block, is_cifar_model, N):
    if is_cifar_model:
        re_NAME = re_NAME_CIFAR
        fmt_NAME = fmt_NAME_CIFAR
        groups = group_CIFAR
    else:
        re_NAME = re_NAME_IMAGENET
        fmt_NAME = fmt_NAME_IMAGENET
        groups = group_IMAGENET
    structure = get_structure(is_cifar_model, N)

    name_mapping = {}
    discarded_dict = {}
    discard_first_block = dict(zip(groups, [0] * len(groups)))
    for i in groups:
        discarded_dict.setdefault(i, [])
    for name in discarded_block:
        rst = re.search(re_NAME, name, re.IGNORECASE)
        if rst:
            group = int(rst.group(1))
            block = int(rst.group(2))
            discarded_dict[group].append(block)
     
    for k,v in discarded_dict.iteritems():
        print('k={}, v={}'.format(k,v))
        print('sturcture={}'.format(structure))
        kept_block = sorted([i for i in range(structure[k]) if i not in v])
        kept_idx = range(len(kept_block))
        old_names = [fmt_NAME%(k, kept_block[i]) for i in kept_idx]
        #if res{}.0 or group{}/block0 is discarded,don't overwrite the block name 
        #   When building the compressed ResNet model, the block is created only to increase the dimension.
        if 0 in kept_block:
            new_names = [fmt_NAME%(k, i) for i in kept_idx]
        else:
            new_names = [fmt_NAME%(k, i+1) for i in kept_idx]
            discard_first_block[k] = 1
        name_mapping.update(dict(zip(old_names, new_names)))
        structure[k] = len(kept_block)
        print("#block in group%d: %d"%(k, len(kept_block)))
    print('name_mapping: {}'.format(name_mapping))
    print("The first block is discarded: {}".format(discard_first_block))
    if is_cifar_model:
        structure = structure[1:]
    return name_mapping, discard_first_block, structure

def gen_cfg(is_cifar_model, model_dir, N, discarded_block, step, discard_first_block, structure, val_error):
    cfg_path = (fmt_saved_model + '.cfg')%(model_dir,step)
    model_path = (fmt_saved_model + '.data-00000-of-00001')%(model_dir,step)
    model_path = os.path.abspath(model_path)
    first_block_flag = []
    for k in sorted(discard_first_block.keys()):
        first_block_flag.append(discard_first_block[k])
    with open(cfg_path,'w') as f:
        f.write('N: {}\n'.format(N))
        f.write('model: {}\n'.format(model_path))
        f.write('step: {}\n'.format(step))
        f.write('discarded_block: {}\n'.format(discarded_block))
        f.write('is_cifar_model: {}\n'.format(is_cifar_model))
        f.write('discard_first_block: {}\n'.format(first_block_flag))
        f.write('structure: {}\n'.format(structure))
        f.write('val_error: {}\n'.format(val_error))

def read_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        for l in f:
            if l.startswith('N: '):
                N = int(l.strip().split(' ')[1])
            if l.startswith('discard_first_block: ['):
                arr = l.replace('discard_first_block: [','').strip(']\n').split(', ')
                discard_first_block = [int(i) for i in arr]
            if l.startswith('structure: ['):
                structure = l.replace('structure: [','').strip(']\n').split(', ')
                structure = [int(i) for i in structure]
            if l.startswith('model: '):
                model_path = l.strip().replace('model: ', '')
    print('N={}'.format(N))
    print('structure={}'.format(structure))
    print('discard_first_block={}'.format(discard_first_block))
    print('model_path={}'.format(model_path))
    return N, structure, discard_first_block, model_path

def compress(is_cifar_model, model_dir, step, name_mapping):
    vars = tf.contrib.framework.list_variables(model_dir)
    with tf.Graph().as_default(), tf.Session().as_default() as sess:
        new_vars = []
        for name, shape in vars:
            #print('----------------')
            #print('old name:{}'.format(name))
            v = tf.contrib.framework.load_variable(model_dir, name)
            prefix = name.split('/')[0]
            if 'tower' not in name :
                # 'convshortcut': a conv to increase dimension for ImagNet
                if prefix in kept_variable or 'convshortcut' in name:
                    new_vars.append(tf.Variable(v, name=name))
                    #print("no change in name:{}".format(new_vars[-1].name))
                else:
                    blk = get_block_name(is_cifar_model, name)  
                    if blk in name_mapping: 
                        new_vars.append(tf.Variable(v, name=name.replace(blk, name_mapping[blk])))
                        #print("new name:{}".format(new_vars[-1].name))
                    else:
                        #print('{} is discarded'.format(name))
                        pass
            else:
                #print('{} is discarded'.format(name))
                pass 
        saver = tf.train.Saver(new_vars)
        sess.run(tf.global_variables_initializer())
        saved_path = fmt_saved_model%(model_dir,step)
        saver.save(sess, saved_path)
        print('The model is compressed and saved at {}'.format(saved_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help="saved model directory", type=str, required=True)
    parser.add_argument('--step', help="which step of model is compressed ", type=int, required=True)
    args = parser.parse_args()
    model_dir, is_cifar, name_mapping = setup(args.dir, args.step)
    compress(is_cifar, model_dir, args.step, name_mapping)
    clear(model_dir)
