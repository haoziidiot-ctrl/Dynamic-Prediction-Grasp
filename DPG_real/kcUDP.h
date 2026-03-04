#pragma once
#ifndef KCUDP_H
#define KCUDP_H
#include <QUdpSocket>
#include <QHostAddress>
#include <QDebug>
#include <QTime>
#include <QKeyEvent>
#include <QThread>
#include <cmath>
#include <QTimer>
// 报文类型枚举
enum AGVMessageType : uint8_t {
	REQUEST = 0x00,  // 请求报文
	RESPONSE = 0x01   // 应答报文
};
//执行码
enum AGVExecuteType : uint8_t {
	ExecuteREQUEST = 0x00,  // 请求报文
	ExecuteRESPONSE = 0x01,   // 应答报文

	Reserved1 = 0x00
};
enum RobotControlCommands : uint8_t {
	Version = 0x01,//默认协议版本号0x01
	ServiceM = 0x10,//服务码 固定0x10
	ORDER_MODE_manual = 0x03,//更改变量
	ORDER_MODE_SWITCH = 0x11,  // 手自动切换
	ORDER_ROBOT_MANUAL_POSITION = 0x14,  // 机器人手动定位
	ORDER_GET_ROBOT_POSITION = 0x15,  // 获取机器人当前位置
	ORDER_NAVIGATION_CONTROL = 0x16,  // 导航控制
	ORDER_QUERY_ROBOT_STATUS = 0x17,  // 查询机器人运行状态
	ORDER_QUERY_NAVIGATION_STATUS = 0x1D,  // 查询机器人导航状态
	ORDER_CONFIRM_ROBOT_POSITION = 0x1F,  // 确认机器人位置
	ORDER_CONFIRM_ROBOT_STATUS = 0xAF  //查询机器人位置
};
enum RobotControlReturn : uint8_t {
	SuccessM = 0x00,//执行成功
	ErroM = 0x01,//执行失败，原因未知
	ServerM = 0x02,//服务码错误
	OrderM = 0x03,  //命令码错误
	HeaderM = 0x04,  //报文头错误
	MessageL = 0x05,  //报文长度错误
	BeyondLimit = 0x06  //路径点超限
	//路径偏移不匹配
	//拼接路径序列号不匹配
	//拼接任务序列号不匹配
	//任务拼接超过最大任务数
};
enum ControlPhase {
	PHASE_CHECK_STATUS,   // 查询状态阶段
	PHASE_GET_LOCATION,   // 查询位置阶段
	PHASE_CONTROL_LOGIC   // 控制逻辑阶段
};
struct RobotData {
	QByteArray robotKey = NULL;
	quint8 protocolVersion = 0;
	quint8 messageType = 0;
	quint16 communicationSequence = 0;
	quint8 serviceCode = 0;
	quint8 commandCode = 0;
	quint8 executeCode = 0;
	quint16 reserved1 = 0;
	quint16 messageLength = 0;
	quint16 reserved2 = 0;
	double x = 0;
	double y = 0;
	double angle = 0;
	double confidence = 0;
	quint16 totalLasers = 0;
	quint16 validLasers = 0;
	quint16 matchedLasers = 0;
	quint16 matchedColumns = 0;
	quint16 pathPointId = 0;
	quint16 pathId = 0;
	quint16 firstHighAreaId = 0;
	quint16 obstacleGroupNumber = 0;
	double odomX = 0;
	double odomY = 0;
	double odomAngle = 0;
	quint64 currentHighAreaId = 0;

	double bodyTemperature = 0;
	double xCoordinate = 0;
	double yCoordinate = 0;
	double orientationAngle = 0;
	double batteryLevel = 0;
	quint8 isBlocked = 0;
	quint8 isCharging = 0;
	quint8 runMode = 0;
	quint8 mapLoadStatus = 0;
	quint32 currentTargetId = 0;
	double forwardSpeed = 0;
	double turnSpeed = 0;
	double batteryVoltage = 0;
	double current = 0;
	quint8 taskStatus = 0;
	quint8 positioningMode = 0;
	quint16 mapVersion = 0;
	quint32 reserved3 = 0;
	double totalMileage = 0;
	double currentRunTime = 0;
	double totalRunTime = 0;
	quint8 positioningStatus = 0;//定位状态
	quint8 trafficControl = 0;
	quint16 reserved4 = 0;
	quint32 mapCount = 0;
	QString currentMapName = 0;

	quint8 abnormal_size = 0;//异常事件状态信息长度
	quint8 action_size = 0;//动作状态长度
	quint8 Info_size = 0;//信息数量
	float GlobalX = 0;
	float GlobalY = 0;
	float GlobalA = 0;;
	quint32 endID = 0;//最后通过的点
	quint32 endParagraph = 0;//最后通过的段
	quint32 Dotserianumber = 0;
	quint8 confidenceAF = 0;
	float LinearVelocity = 0;//线速度
	float Acceleration = 0;//加速度
	float AngularVelocity = 0;//角速度
	quint8 WorkMode = 0;
	quint8 AGVStatus = 0;
	quint8 AbilitySettings = 0;//能力设置
	quint32 OrderID = 0;
	quint32 OrderKEY = 0;
	quint8 PointStateSequence = 0;//点状态序列
	quint8 PathStateSequence = 0;//段状态序列
	float Batterylevel = 0;//电量百分比
	float VoltageAF = 0;//电压
	float ElectricCurrent = 0;//电流
	quint8 ChargingStatus = 0;
};
class kcUDP : public QObject
{
	Q_OBJECT
public:

	explicit kcUDP(QString& kc_key, QObject* parent = NULL);

	~kcUDP();

	////void init_port();  //初始化串口
	void dealMsg();    //处理对方发过来的数据
	QByteArray minglin;

	void write_datakcUDP(QByteArray str);
	void AAdatakcUDP(bool DATA);

	void ExecuteControlLogic();

	bool m_taskReCall = false;//开始反复执行

public slots:
	void handle_data();
	void Inint();//初始化函数

signals:

	void receive_datakcUDP(QString str);

protected:
	QString ip = "192.168.100.178";//导航地址
	qint16 port = 17804;//导航端口
	QByteArray hexToQByteArray(QString& hexStr);
	QByteArray convertTo16Array(QString& input);
	QByteArray convertToLittleEndian(quint16 value);
	bool isLittleEndian();
	QByteArray toLittleEndian(double value);
	//static RobotData parseResponse(const QByteArray& data);
	//static QString formatData(const RobotData& data);

	RobotData parseResponse(const QByteArray& data);
	QString formatData(const RobotData& data);
	/**************导航初始化部分*****************/
	//切换定位为手动模式（命令码：0x11)//上电默认手动模式
	QByteArray SwitchTOmanual();
	//执行机器人手动定位（命令码：0x14）//手动定位获取其制图后的位置进行定位
	QByteArray ExecuteManually();
	//查询机器人运行状态（命令码：0xAF）
	QByteArray CheckStatus();
	//确认初始位置（命令码：0x1F）
	QByteArray Confirmlocation();
	//切换成自动模式（命令码：0x11）Switch to automatic
	QByteArray SwitchTOautomatic();
	//初始化完成，可以接收导航任务
	/**************导航任务部分*****************/
	//查询机器人运行状态（命令码：0x17）1//地图是否载入成功是其一 ，机器人定位状态是否成功是其二
	QByteArray CheckRunningStatus();
	//切换机器人导航状态（命令码：0x1D）
	QByteArray NavigationStatus();
	//获取机器人当前位置（命令码：0x15）//置信度需要再定位成功后查询，第一次的置信度会有偏差？
	QByteArray GetLocation();
	//导航控制（0x16）
	QByteArray NavigationControl();
	QByteArray NavigationControlBack();
private:

	QUdpSocket* udpSocket;
	QThread* my_thread;
	QTimer* reconnectTimer;

	QByteArray devicekey;//设备密钥
	//quint8 Version = 1;//版本号默认0x01
	//quint8 RequestMessage = 0;//请求报文0x00 应答报文0x01
	//quint8 ResponseMessage = 1;
	quint16 SequenceA = 0;//请求序列号
	quint16 SequenceB = 0x00;//应答序列号

	//quint8 ServiceM = 0x10;//服务码 固定0x10
	//quint8 OrderM = 0x00;//命令码 命令类型
	//quint8 Execute = 0;//执行码，表明命令执行情况，请求时为0
	//quint8 Reserved1 = 0;//预留默认00
	quint16 MessageLenth = 0;//数据长度 不超过512
	quint8 Reserved2[2] = { 0 };
	/*quint8 MessageData[] = 0;*/
	double Tx;
	double Ty;
	double Ta;

	QByteArray MessageData = "";

	bool initUDP = false;//首次开机返回 即使返回成功也要确定位置
	int m_switchModel = 999;//当前手自动模式
	int m_mapLoad = 999;//地图载入是否成功
	//int RobotStaut = 999;//机器人状态

	bool STARTN = false;//开始导航
	double confidenceUDP = 0;//置信度

	bool m_sendLocationNext = false;
	bool m_isExecutingManual = false;//执行手动定位
	bool m_navigating = false;//开始导航
	int m_navigationStatus = 999;//导航状态

	bool SwitchtoA = false;//切换到自动
	bool SwitchtoM = true;//切换到自动

	int m_taskStatus = 999;//任务状态

	bool m_taskStart = false;
	bool m_taskEnd = false;//返回后任务结束

	bool m_controlMode = false;//控制

	QTimer* m_statusTimer;
};

#endif