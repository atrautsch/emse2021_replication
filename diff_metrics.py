import os
import pickle
import numpy as np

from mongoengine import connect
from pycoshark.mongomodels import Project, VCSSystem, Commit, Tag, File, CodeEntityState, FileAction, People, IssueSystem, Issue, Message, MailingList, Event, MynbouData, Identity, Hunk, Branch
from pycoshark.utils import java_filename_filter
local = {'host': 'localhost',
       'port': 27017,
       'db': 'smartshark',
       'username': '',
       'password': '',
       'authentication_source': '',
       'connect': False}
connect(**local)

PROJECTS = ['calcite',
 'cayenne',
 'commons-bcel',
 'commons-beanutils',
 'commons-codec',
 'commons-collections',
 'commons-compress',
 'commons-configuration',
 'commons-dbcp',
 'commons-digester',
 'commons-imaging',
 'commons-io',
 'commons-jcs',
 'commons-jexl',
 'commons-lang',
 'commons-math',
 'commons-net',
 'commons-rdf',
 'commons-scxml',
 'commons-validator',
 'commons-vfs',
 'eagle',
 'falcon',
 'flume',
 'giraph',
 'gora',
 'jspwiki',
 'knox',
 'kylin',
 'lens',
 'mahout',
 'manifoldcf',
 'opennlp',
 'parquet-mr',
 'pdfbox',
 'phoenix',
 'ranger',
 'santuario-java',
 'storm',
 'struts',
 'systemml',
 'tez',
 'tika',
 'wss4j',
 'zeppelin',
 'helix',
  'httpcomponents-client', 'archiva', 'httpcomponents-core', 'jena', 'streams', 'mina-sshd', 'roller', 'nifi']
PROJECTS = sorted(PROJECTS)

def get_metrics(vcs_id, current_hash, file_id):
    agg = {}
    c = Commit.objects.get(revision_hash=current_hash, vcs_system_id=vcs_id)
    try:
        file_ces = CodeEntityState.objects.get(id__in=c.code_entity_states, file_id=file_id, ce_type='file')
    except CodeEntityState.DoesNotExist:
        print('ces not found')
        return agg

    for k, v in file_ces.metrics.items():
        agg[k + '_file'] = v
    
    for w in file_ces.linter:
        key = '{}'.format(w['l_ty'])
        if key not in agg.keys():
            agg[key] = 0
        agg[key] += 1

    for subfile_ces in CodeEntityState.objects.filter(id__in=c.code_entity_states, file_id=file_id):
        if subfile_ces.ce_type == 'file':
            continue

        for k, v in subfile_ces.metrics.items():
            key = k + '_' + subfile_ces.ce_type + '_sum'
            if key not in agg.keys():
                agg[key] = 0
            if not np.isnan(v):
                agg[key] += v

    return agg


def get_file_ids(commit, parent):
    file_ids_current = []
    file_ids_parent = []
    for fa in FileAction.objects.filter(commit_id=commit.id, parent_revision_hash=parent.revision_hash):
        f = File.objects.get(id=fa.file_id)

        if fa.mode.lower() in ['c', 'r']:
            f2 = File.objects.get(id=fa.old_file_id)

        if not java_filename_filter(f.path, production_only=True):
            continue

        # added file, only in new
        if fa.mode.lower() == 'a':
            file_ids_current.append(f.id)

        # deleted, only in old
        elif fa.mode.lower() == 'd':
            file_ids_parent.append(f.id)

        # copy or rename, new version only in current, old in parent
        elif fa.mode.lower() in ['c', 'r']:
            file_ids_parent.append(f2.id)
            file_ids_current.append(f.id)

        # anything else (modified) same in both
        else:
            file_ids_current.append(f.id)
            file_ids_parent.append(f.id)
    return file_ids_current, file_ids_parent


def get_metrics2(vcs_id, current_hash, file_id):
    agg = {}
    c = Commit.objects.get(revision_hash=current_hash, vcs_system_id=vcs_id)
    try:
        file_ces = CodeEntityState.objects.get(id__in=c.code_entity_states, file_id=file_id, ce_type='file')
    except CodeEntityState.DoesNotExist:
        print('ces not found')
        return agg

    for k, v in file_ces.metrics.items():
        agg[k + '_file'] = v

    for subfile_ces in CodeEntityState.objects.filter(id__in=c.code_entity_states, file_id=file_id):
        if subfile_ces.ce_type == 'file':
            continue

        for k, v in subfile_ces.metrics.items():
            key = k + '_' + subfile_ces.ce_type + '_sum'
            if key not in agg.keys():
                agg[key] = 0
            if not np.isnan(v):
                agg[key] += v

    return agg

def get_diff_metrics(vcs_id, current_hash, parent_hash, file_id):
    agg_current = get_metrics(vcs_id, current_hash, file_id)
    agg_parent = get_metrics(vcs_id, parent_hash, file_id)

    agg = {}
    for k, v in agg_current.items():
        agg[k] = v
        if k in agg_parent.keys() and not np.isnan(agg_parent[k]):
            agg[k] -= agg_parent[k]
        else:
            # this happens when e.g., a enum is added in the current commit that was not there in the parent    
            print('{} not in file ({}) commit ({}) current ({})'.format(k, file_id, parent_hash, current_hash))
            # del agg[k]
            # print(k, 'not in parent', len(agg_parent.keys()))

    return agg


def get_system_agg(vcs_id, current_hash):
    agg = {}
    c = Commit.objects.get(revision_hash=current_hash, vcs_system_id=vcs_id)

    for file_ces in CodeEntityState.objects.filter(id__in=c.code_entity_states, ce_type='file').timeout(False):
        f = File.objects.get(id=file_ces.file_id)
        if not java_filename_filter(f.path, production_only=True):
            continue
        
        num_files = 0
        for k, v in file_ces.metrics.items():
            key = 'system_{}_file'.format(k)
            if key not in agg.keys():
                agg[key] = 0
            agg[key] += v
    
        for w in file_ces.linter:
            key = 'system_{}'.format(w['l_ty'])
            if key not in agg.keys():
                agg[key] = 0
            agg[key] += 1

        for subfile_ces in CodeEntityState.objects.filter(id__in=c.code_entity_states, file_id=f.id):
            counter = 'system_num_{}'.format(subfile_ces.ce_type)
            if counter not in agg.keys():
                agg[counter] = 0
            agg[counter] += 1

            if subfile_ces.ce_type == 'file':
                continue

            for k, v in subfile_ces.metrics.items():
                key = 'system_' + k + '_' + subfile_ces.ce_type + '_sum'
                if key not in agg.keys():
                    agg[key] = 0
                if not np.isnan(v):
                    agg[key] += v

    return agg

# 1. get candidate commits from all projects
# PROJECTS = ['commons-validator']


all_commits = 0
for project in PROJECTS:
    fname = '{}.metrics_allchanges.pickle'.format(project)

    if os.path.exists(fname):
        continue

    list_commits = []
    p = Project.objects.get(name=project)
    vcs = VCSSystem.objects.get(project_id=p.id)
    all_commits += Commit.objects.filter(vcs_system_id=vcs.id).count()
    commits = [(c[0], c[1], c[2], c[3], c[4]) for c in Commit.objects.filter(vcs_system_id=vcs.id).values_list('revision_hash', 'committer_date', 'message', 'id', 'parents')]

    for commit in commits:
        # skip merge commits
        if len(commit[4]) > 1:
            continue
        
        # skip orphan commits
        if len(commit[4]) == 0:
            continue

        current_hash = commit[0]
        parent_hash = commit[4][0]

        # remove lines that do not add (intent) information
        message_lines = []
        found_text = False
        for line in commit[2].split('\n'):
            if line.lower().startswith('signed-off-by'):
                continue
            if line.lower().startswith('git-svn-id'):
                continue
            if '*** empty log message ***' in line.lower():
                continue
            if line.strip():
                found_text = True

            message_lines.append(line)
            
        # skip empty commit messages
        if not found_text:
            continue
        
        message = '\n'.join(message_lines)

        # restrict commits to changes where java production files have been changed
        java_changed = False
        metric_agg = {'lines_added': 0, 'lines_deleted': 0, 'files_modified': 0, 'num_hunks': 0}
        for fa in FileAction.objects.filter(commit_id=commit[3]):
            
            # only modified files
            #if fa.mode.lower() != 'm':
            #    continue

            # only files that are production and java
            f = File.objects.get(id=fa.file_id)
            if java_filename_filter(f.path, production_only=True):
                file_id = f.id
                metric_agg['lines_added'] += fa.lines_added
                metric_agg['lines_deleted'] += fa.lines_deleted
                metric_agg['files_modified'] += 1
                metric_agg['num_hunks'] += Hunk.objects.filter(file_action_id=fa.id).count()
                java_changed = True
    
                # get aggregated metric changes between files in both current and parent
                # m = get_diff_metrics(vcs.id, commit[0], commit[4][0], f.id)
                agg_current = get_metrics(vcs.id, current_hash, file_id)

                # in case of rename we need to use the previous file id
                if fa.old_file_id:
                    f2 = File.objects.get(id=fa.old_file_id)
                    file_id = f2.id
                agg_parent = get_metrics(vcs.id, parent_hash, file_id)

                # deltas aggregated
                agg = {}
                for k, v in agg_current.items():
                    agg[k] = v
                    if k in agg_parent.keys() and not np.isnan(agg_parent[k]):
                        agg[k] -= agg_parent[k]
                    else:
                        # this happens when e.g., a enum is added in the current commit that was not there in the parent
                        pass

                for k, v in agg_current.items():
                    ckey = 'current_{}'.format(k)
                    if ckey not in metric_agg.keys():
                        metric_agg[ckey] = 0
                    metric_agg[ckey] += v

                for k, v in agg_parent.items():
                    pkey = 'parent_{}'.format(k)
                    if pkey not in metric_agg.keys():
                        metric_agg[pkey] = 0
                    metric_agg[pkey] += v
                        
                for k, v in agg.items():
                    dkey = 'delta_{}'.format(k)
                    if dkey not in metric_agg.keys():
                        metric_agg[dkey] = 0

                    if not np.isnan(v):
                        metric_agg[dkey] += v

        if java_changed:
            tmp = {'project': project, 'revision_hash': commit[0], 'committer_date': commit[1], 'message': message}
            tmp.update(metric_agg)
            
            #system_agg = get_system_agg(vcs.id, current_hash)
            #tmp.update(system_agg)
            list_commits.append(tmp)

    with open('{}.metrics3.pickle'.format(project), 'wb') as f:
        pickle.dump(list_commits, f)
print(all_commits)
