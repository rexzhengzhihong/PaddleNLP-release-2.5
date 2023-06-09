from pprint import pprint
from paddlenlp import Taskflow
from paddlenlp.utils.doc_parser import DocParser


def zzszyfptest():
    schema = ['开票日期', '名称', '纳税人识别号', '开户行及账号', '金额', '价税合计', 'No', '税率', '地址、电话', '税额']
    my_ie = Taskflow("information_extraction", model="uie-x-base", schema=schema,
                     task_path='/home/DiskA/zncsPython/picture_uie/zzszyfp_test/checkpoint/model_best', precison='fp16',
                     layout_analysis=True)
    doc_path = "/home/DiskA/zncsPython/picture_uie/zzszyfp_test/data/images/b3.jpg"
    results = my_ie({"doc": doc_path})
    pprint(results)

    # 结果可视化
    DocParser.write_image_with_results(doc_path, result=results[0],
                                       save_path="/home/DiskA/zncsPython/picture_uie/zzszyfp_test/data/image_show.png")


def zzszyfp_v1():
    schema = ["购买方名称", "购买方纳税人识别号", "购买方地址电话", "购买方开户行及账号", "合计金额", "合计税额", "价税合计大写", "价税合计小写", "开票人", "开票日期", "发票号",
              "发票代码", "销售方名称", "销售方纳税人识别号"
                               "销售方地址电话", "销售方开户行及账号", "项目名称", "单位", "数量", "单价", "金额", "税率", "税额",
              {"Description": ["单位", "数量", "单价", "金额", "税率", "税额"]}]

    my_ie = Taskflow("information_extraction", model="uie-x-base", schema=schema,
                     task_path='/home/DiskA/zncsPython/picture_uie/zzszyfp_v1/checkpoint/model_best', precison='fp16',
                     layout_analysis=True)
    doc_path = "/home/DiskA/zncsPython/picture_uie/zzszyfp_v1/data/images/zzszyfp.jpg"
    results = my_ie({"doc": doc_path})
    pprint(results)

    # 结果可视化
    DocParser.write_image_with_results(doc_path, result=results[0],
                                       save_path="/home/DiskA/zncsPython/picture_uie/zzszyfp_v1/data/image_show.png")


if __name__ == '__main__':
    zzszyfptest()
