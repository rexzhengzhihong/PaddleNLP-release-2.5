from pprint import pprint
from paddlenlp import Taskflow


if __name__ == '__main__':
    # 海关报关单信息抽取
    # schema = ["收发货人", "进口口岸", "进口日期", "运输方式", "征免性质", "境内目的地", "运输工具名称", "包装种类", "件数", "合同协议号"]
    # ie = Taskflow("information_extraction", schema=schema, model="uie-x-base")
    # pprint(ie({"doc": "../input/custom.jpeg"}))

    # 关系抽取
    # schema = {"姓名": ["招聘单位", "报考岗位"]}
    # ie = Taskflow("information_extraction", schema=schema, model="uie-x-base")
    # pprint(ie({"doc": "../input/table.png"}))

    ### 跨任务抽取，同时进行实体、关系抽取
    schema = ["Total GBP", "No.", "Date", "Customer No.", "Subtotal without VAT",
              {"Description": ["Quantity", "Amount"]}]
    ie = Taskflow("information_extraction", schema=schema, model="uie-x-base", ocr_lang="en", schema_lang="en")
    pprint(ie({"doc": "../input/delivery_note.png"}))